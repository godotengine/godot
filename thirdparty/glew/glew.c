/*
** The OpenGL Extension Wrangler Library
** Copyright (C) 2008-2015, Nigel Stewart <nigels[]users sourceforge net>
** Copyright (C) 2002-2008, Milan Ikits <milan ikits[]ieee org>
** Copyright (C) 2002-2008, Marcelo E. Magallon <mmagallo[]debian org>
** Copyright (C) 2002, Lev Povalahev
** All rights reserved.
** 
** Redistribution and use in source and binary forms, with or without 
** modification, are permitted provided that the following conditions are met:
** 
** * Redistributions of source code must retain the above copyright notice, 
**   this list of conditions and the following disclaimer.
** * Redistributions in binary form must reproduce the above copyright notice, 
**   this list of conditions and the following disclaimer in the documentation 
**   and/or other materials provided with the distribution.
** * The name of the author may be used to endorse or promote products 
**   derived from this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
** ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
** LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
** CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
** SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
** INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
** CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
** ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
** THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <GL/glew.h>

#if defined(_WIN32)
#  include <GL/wglew.h>
#elif !defined(__ANDROID__) && !defined(__native_client__) && !defined(__HAIKU__) && (!defined(__APPLE__) || defined(GLEW_APPLE_GLX))
#  include <GL/glxew.h>
#endif

#include <stddef.h>  /* For size_t */

/*
 * Define glewGetContext and related helper macros.
 */
#ifdef GLEW_MX
#  define glewGetContext() ctx
#  ifdef _WIN32
#    define GLEW_CONTEXT_ARG_DEF_INIT GLEWContext* ctx
#    define GLEW_CONTEXT_ARG_VAR_INIT ctx
#    define wglewGetContext() ctx
#    define WGLEW_CONTEXT_ARG_DEF_INIT WGLEWContext* ctx
#    define WGLEW_CONTEXT_ARG_DEF_LIST WGLEWContext* ctx
#  else /* _WIN32 */
#    define GLEW_CONTEXT_ARG_DEF_INIT void
#    define GLEW_CONTEXT_ARG_VAR_INIT
#    define glxewGetContext() ctx
#    define GLXEW_CONTEXT_ARG_DEF_INIT void
#    define GLXEW_CONTEXT_ARG_DEF_LIST GLXEWContext* ctx
#  endif /* _WIN32 */
#  define GLEW_CONTEXT_ARG_DEF_LIST GLEWContext* ctx
#else /* GLEW_MX */
#  define GLEW_CONTEXT_ARG_DEF_INIT void
#  define GLEW_CONTEXT_ARG_VAR_INIT
#  define GLEW_CONTEXT_ARG_DEF_LIST void
#  define WGLEW_CONTEXT_ARG_DEF_INIT void
#  define WGLEW_CONTEXT_ARG_DEF_LIST void
#  define GLXEW_CONTEXT_ARG_DEF_INIT void
#  define GLXEW_CONTEXT_ARG_DEF_LIST void
#endif /* GLEW_MX */

#if defined(GLEW_REGAL)

/* In GLEW_REGAL mode we call direcly into the linked
   libRegal.so glGetProcAddressREGAL for looking up
   the GL function pointers. */

#  undef glGetProcAddressREGAL
#  ifdef WIN32
extern void *  __stdcall glGetProcAddressREGAL(const GLchar *name);
static void * (__stdcall * regalGetProcAddress) (const GLchar *) = glGetProcAddressREGAL;
#    else
extern void * glGetProcAddressREGAL(const GLchar *name);
static void * (*regalGetProcAddress) (const GLchar *) = glGetProcAddressREGAL;
#  endif
#  define glGetProcAddressREGAL GLEW_GET_FUN(__glewGetProcAddressREGAL)

#elif defined(__sgi) || defined (__sun) || defined(__HAIKU__) || defined(GLEW_APPLE_GLX)
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

void* dlGetProcAddress (const GLubyte* name)
{
  static void* h = NULL;
  static void* gpa;

  if (h == NULL)
  {
    if ((h = dlopen(NULL, RTLD_LAZY | RTLD_LOCAL)) == NULL) return NULL;
    gpa = dlsym(h, "glXGetProcAddress");
  }

  if (gpa != NULL)
    return ((void*(*)(const GLubyte*))gpa)(name);
  else
    return dlsym(h, (const char*)name);
}
#endif /* __sgi || __sun || GLEW_APPLE_GLX */

#if defined(__APPLE__)
#include <stdlib.h>
#include <string.h>
#include <AvailabilityMacros.h>

#ifdef MAC_OS_X_VERSION_10_3

#include <dlfcn.h>

void* NSGLGetProcAddress (const GLubyte *name)
{
  static void* image = NULL;
  void* addr;
  if (NULL == image) 
  {
    image = dlopen("/System/Library/Frameworks/OpenGL.framework/Versions/Current/OpenGL", RTLD_LAZY);
  }
  if( !image ) return NULL;
  addr = dlsym(image, (const char*)name);
  if( addr ) return addr;
#ifdef GLEW_APPLE_GLX
  return dlGetProcAddress( name ); // try next for glx symbols
#else
  return NULL;
#endif
}
#else

#include <mach-o/dyld.h>

void* NSGLGetProcAddress (const GLubyte *name)
{
  static const struct mach_header* image = NULL;
  NSSymbol symbol;
  char* symbolName;
  if (NULL == image)
  {
    image = NSAddImage("/System/Library/Frameworks/OpenGL.framework/Versions/Current/OpenGL", NSADDIMAGE_OPTION_RETURN_ON_ERROR);
  }
  /* prepend a '_' for the Unix C symbol mangling convention */
  symbolName = malloc(strlen((const char*)name) + 2);
  strcpy(symbolName+1, (const char*)name);
  symbolName[0] = '_';
  symbol = NULL;
  /* if (NSIsSymbolNameDefined(symbolName))
	 symbol = NSLookupAndBindSymbol(symbolName); */
  symbol = image ? NSLookupSymbolInImage(image, symbolName, NSLOOKUPSYMBOLINIMAGE_OPTION_BIND | NSLOOKUPSYMBOLINIMAGE_OPTION_RETURN_ON_ERROR) : NULL;
  free(symbolName);
  if( symbol ) return NSAddressOfSymbol(symbol);
#ifdef GLEW_APPLE_GLX
  return dlGetProcAddress( name ); // try next for glx symbols
#else
  return NULL;
#endif
}
#endif /* MAC_OS_X_VERSION_10_3 */
#endif /* __APPLE__ */

/*
 * Define glewGetProcAddress.
 */
#if defined(GLEW_REGAL)
#  define glewGetProcAddress(name) regalGetProcAddress((const GLchar *) name)
#elif defined(_WIN32)
#  define glewGetProcAddress(name) wglGetProcAddress((LPCSTR)name)
#elif defined(__APPLE__) && !defined(GLEW_APPLE_GLX)
#  define glewGetProcAddress(name) NSGLGetProcAddress(name)
#elif defined(__sgi) || defined(__sun) || defined(__HAIKU__)
#  define glewGetProcAddress(name) dlGetProcAddress(name)
#elif defined(__ANDROID__)
#  define glewGetProcAddress(name) NULL /* TODO */
#elif defined(__native_client__)
#  define glewGetProcAddress(name) NULL /* TODO */
#else /* __linux */
#  define glewGetProcAddress(name) (*glXGetProcAddressARB)(name)
#endif

/*
 * Redefine GLEW_GET_VAR etc without const cast
 */

#undef GLEW_GET_VAR
#ifdef GLEW_MX
# define GLEW_GET_VAR(x) (glewGetContext()->x)
#else /* GLEW_MX */
# define GLEW_GET_VAR(x) (x)
#endif /* GLEW_MX */

#ifdef WGLEW_GET_VAR
# undef WGLEW_GET_VAR
# ifdef GLEW_MX
#  define WGLEW_GET_VAR(x) (wglewGetContext()->x)
# else /* GLEW_MX */
#  define WGLEW_GET_VAR(x) (x)
# endif /* GLEW_MX */
#endif /* WGLEW_GET_VAR */

#ifdef GLXEW_GET_VAR
# undef GLXEW_GET_VAR
# ifdef GLEW_MX
#  define GLXEW_GET_VAR(x) (glxewGetContext()->x)
# else /* GLEW_MX */
#  define GLXEW_GET_VAR(x) (x)
# endif /* GLEW_MX */
#endif /* GLXEW_GET_VAR */

/*
 * GLEW, just like OpenGL or GLU, does not rely on the standard C library.
 * These functions implement the functionality required in this file.
 */
static GLuint _glewStrLen (const GLubyte* s)
{
  GLuint i=0;
  if (s == NULL) return 0;
  while (s[i] != '\0') i++;
  return i;
}

static GLuint _glewStrCLen (const GLubyte* s, GLubyte c)
{
  GLuint i=0;
  if (s == NULL) return 0;
  while (s[i] != '\0' && s[i] != c) i++;
  return (s[i] == '\0' || s[i] == c) ? i : 0;
}

static GLboolean _glewStrSame (const GLubyte* a, const GLubyte* b, GLuint n)
{
  GLuint i=0;
  if(a == NULL || b == NULL)
    return (a == NULL && b == NULL && n == 0) ? GL_TRUE : GL_FALSE;
  while (i < n && a[i] != '\0' && b[i] != '\0' && a[i] == b[i]) i++;
  return i == n ? GL_TRUE : GL_FALSE;
}

static GLboolean _glewStrSame1 (const GLubyte** a, GLuint* na, const GLubyte* b, GLuint nb)
{
  while (*na > 0 && (**a == ' ' || **a == '\n' || **a == '\r' || **a == '\t'))
  {
    (*a)++;
    (*na)--;
  }
  if(*na >= nb)
  {
    GLuint i=0;
    while (i < nb && (*a)+i != NULL && b+i != NULL && (*a)[i] == b[i]) i++;
    if(i == nb)
    {
      *a = *a + nb;
      *na = *na - nb;
      return GL_TRUE;
    }
  }
  return GL_FALSE;
}

static GLboolean _glewStrSame2 (const GLubyte** a, GLuint* na, const GLubyte* b, GLuint nb)
{
  if(*na >= nb)
  {
    GLuint i=0;
    while (i < nb && (*a)+i != NULL && b+i != NULL && (*a)[i] == b[i]) i++;
    if(i == nb)
    {
      *a = *a + nb;
      *na = *na - nb;
      return GL_TRUE;
    }
  }
  return GL_FALSE;
}

static GLboolean _glewStrSame3 (const GLubyte** a, GLuint* na, const GLubyte* b, GLuint nb)
{
  if(*na >= nb)
  {
    GLuint i=0;
    while (i < nb && (*a)+i != NULL && b+i != NULL && (*a)[i] == b[i]) i++;
    if (i == nb && (*na == nb || (*a)[i] == ' ' || (*a)[i] == '\n' || (*a)[i] == '\r' || (*a)[i] == '\t'))
    {
      *a = *a + nb;
      *na = *na - nb;
      return GL_TRUE;
    }
  }
  return GL_FALSE;
}

/*
 * Search for name in the extensions string. Use of strstr()
 * is not sufficient because extension names can be prefixes of
 * other extension names. Could use strtok() but the constant
 * string returned by glGetString might be in read-only memory.
 */
static GLboolean _glewSearchExtension (const char* name, const GLubyte *start, const GLubyte *end)
{
  const GLubyte* p;
  GLuint len = _glewStrLen((const GLubyte*)name);
  p = start;
  while (p < end)
  {
    GLuint n = _glewStrCLen(p, ' ');
    if (len == n && _glewStrSame((const GLubyte*)name, p, n)) return GL_TRUE;
    p += n+1;
  }
  return GL_FALSE;
}

#if !defined(_WIN32) || !defined(GLEW_MX)

PFNGLCOPYTEXSUBIMAGE3DPROC __glewCopyTexSubImage3D = NULL;
PFNGLDRAWRANGEELEMENTSPROC __glewDrawRangeElements = NULL;
PFNGLTEXIMAGE3DPROC __glewTexImage3D = NULL;
PFNGLTEXSUBIMAGE3DPROC __glewTexSubImage3D = NULL;

PFNGLACTIVETEXTUREPROC __glewActiveTexture = NULL;
PFNGLCLIENTACTIVETEXTUREPROC __glewClientActiveTexture = NULL;
PFNGLCOMPRESSEDTEXIMAGE1DPROC __glewCompressedTexImage1D = NULL;
PFNGLCOMPRESSEDTEXIMAGE2DPROC __glewCompressedTexImage2D = NULL;
PFNGLCOMPRESSEDTEXIMAGE3DPROC __glewCompressedTexImage3D = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC __glewCompressedTexSubImage1D = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC __glewCompressedTexSubImage2D = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC __glewCompressedTexSubImage3D = NULL;
PFNGLGETCOMPRESSEDTEXIMAGEPROC __glewGetCompressedTexImage = NULL;
PFNGLLOADTRANSPOSEMATRIXDPROC __glewLoadTransposeMatrixd = NULL;
PFNGLLOADTRANSPOSEMATRIXFPROC __glewLoadTransposeMatrixf = NULL;
PFNGLMULTTRANSPOSEMATRIXDPROC __glewMultTransposeMatrixd = NULL;
PFNGLMULTTRANSPOSEMATRIXFPROC __glewMultTransposeMatrixf = NULL;
PFNGLMULTITEXCOORD1DPROC __glewMultiTexCoord1d = NULL;
PFNGLMULTITEXCOORD1DVPROC __glewMultiTexCoord1dv = NULL;
PFNGLMULTITEXCOORD1FPROC __glewMultiTexCoord1f = NULL;
PFNGLMULTITEXCOORD1FVPROC __glewMultiTexCoord1fv = NULL;
PFNGLMULTITEXCOORD1IPROC __glewMultiTexCoord1i = NULL;
PFNGLMULTITEXCOORD1IVPROC __glewMultiTexCoord1iv = NULL;
PFNGLMULTITEXCOORD1SPROC __glewMultiTexCoord1s = NULL;
PFNGLMULTITEXCOORD1SVPROC __glewMultiTexCoord1sv = NULL;
PFNGLMULTITEXCOORD2DPROC __glewMultiTexCoord2d = NULL;
PFNGLMULTITEXCOORD2DVPROC __glewMultiTexCoord2dv = NULL;
PFNGLMULTITEXCOORD2FPROC __glewMultiTexCoord2f = NULL;
PFNGLMULTITEXCOORD2FVPROC __glewMultiTexCoord2fv = NULL;
PFNGLMULTITEXCOORD2IPROC __glewMultiTexCoord2i = NULL;
PFNGLMULTITEXCOORD2IVPROC __glewMultiTexCoord2iv = NULL;
PFNGLMULTITEXCOORD2SPROC __glewMultiTexCoord2s = NULL;
PFNGLMULTITEXCOORD2SVPROC __glewMultiTexCoord2sv = NULL;
PFNGLMULTITEXCOORD3DPROC __glewMultiTexCoord3d = NULL;
PFNGLMULTITEXCOORD3DVPROC __glewMultiTexCoord3dv = NULL;
PFNGLMULTITEXCOORD3FPROC __glewMultiTexCoord3f = NULL;
PFNGLMULTITEXCOORD3FVPROC __glewMultiTexCoord3fv = NULL;
PFNGLMULTITEXCOORD3IPROC __glewMultiTexCoord3i = NULL;
PFNGLMULTITEXCOORD3IVPROC __glewMultiTexCoord3iv = NULL;
PFNGLMULTITEXCOORD3SPROC __glewMultiTexCoord3s = NULL;
PFNGLMULTITEXCOORD3SVPROC __glewMultiTexCoord3sv = NULL;
PFNGLMULTITEXCOORD4DPROC __glewMultiTexCoord4d = NULL;
PFNGLMULTITEXCOORD4DVPROC __glewMultiTexCoord4dv = NULL;
PFNGLMULTITEXCOORD4FPROC __glewMultiTexCoord4f = NULL;
PFNGLMULTITEXCOORD4FVPROC __glewMultiTexCoord4fv = NULL;
PFNGLMULTITEXCOORD4IPROC __glewMultiTexCoord4i = NULL;
PFNGLMULTITEXCOORD4IVPROC __glewMultiTexCoord4iv = NULL;
PFNGLMULTITEXCOORD4SPROC __glewMultiTexCoord4s = NULL;
PFNGLMULTITEXCOORD4SVPROC __glewMultiTexCoord4sv = NULL;
PFNGLSAMPLECOVERAGEPROC __glewSampleCoverage = NULL;

PFNGLBLENDCOLORPROC __glewBlendColor = NULL;
PFNGLBLENDEQUATIONPROC __glewBlendEquation = NULL;
PFNGLBLENDFUNCSEPARATEPROC __glewBlendFuncSeparate = NULL;
PFNGLFOGCOORDPOINTERPROC __glewFogCoordPointer = NULL;
PFNGLFOGCOORDDPROC __glewFogCoordd = NULL;
PFNGLFOGCOORDDVPROC __glewFogCoorddv = NULL;
PFNGLFOGCOORDFPROC __glewFogCoordf = NULL;
PFNGLFOGCOORDFVPROC __glewFogCoordfv = NULL;
PFNGLMULTIDRAWARRAYSPROC __glewMultiDrawArrays = NULL;
PFNGLMULTIDRAWELEMENTSPROC __glewMultiDrawElements = NULL;
PFNGLPOINTPARAMETERFPROC __glewPointParameterf = NULL;
PFNGLPOINTPARAMETERFVPROC __glewPointParameterfv = NULL;
PFNGLPOINTPARAMETERIPROC __glewPointParameteri = NULL;
PFNGLPOINTPARAMETERIVPROC __glewPointParameteriv = NULL;
PFNGLSECONDARYCOLOR3BPROC __glewSecondaryColor3b = NULL;
PFNGLSECONDARYCOLOR3BVPROC __glewSecondaryColor3bv = NULL;
PFNGLSECONDARYCOLOR3DPROC __glewSecondaryColor3d = NULL;
PFNGLSECONDARYCOLOR3DVPROC __glewSecondaryColor3dv = NULL;
PFNGLSECONDARYCOLOR3FPROC __glewSecondaryColor3f = NULL;
PFNGLSECONDARYCOLOR3FVPROC __glewSecondaryColor3fv = NULL;
PFNGLSECONDARYCOLOR3IPROC __glewSecondaryColor3i = NULL;
PFNGLSECONDARYCOLOR3IVPROC __glewSecondaryColor3iv = NULL;
PFNGLSECONDARYCOLOR3SPROC __glewSecondaryColor3s = NULL;
PFNGLSECONDARYCOLOR3SVPROC __glewSecondaryColor3sv = NULL;
PFNGLSECONDARYCOLOR3UBPROC __glewSecondaryColor3ub = NULL;
PFNGLSECONDARYCOLOR3UBVPROC __glewSecondaryColor3ubv = NULL;
PFNGLSECONDARYCOLOR3UIPROC __glewSecondaryColor3ui = NULL;
PFNGLSECONDARYCOLOR3UIVPROC __glewSecondaryColor3uiv = NULL;
PFNGLSECONDARYCOLOR3USPROC __glewSecondaryColor3us = NULL;
PFNGLSECONDARYCOLOR3USVPROC __glewSecondaryColor3usv = NULL;
PFNGLSECONDARYCOLORPOINTERPROC __glewSecondaryColorPointer = NULL;
PFNGLWINDOWPOS2DPROC __glewWindowPos2d = NULL;
PFNGLWINDOWPOS2DVPROC __glewWindowPos2dv = NULL;
PFNGLWINDOWPOS2FPROC __glewWindowPos2f = NULL;
PFNGLWINDOWPOS2FVPROC __glewWindowPos2fv = NULL;
PFNGLWINDOWPOS2IPROC __glewWindowPos2i = NULL;
PFNGLWINDOWPOS2IVPROC __glewWindowPos2iv = NULL;
PFNGLWINDOWPOS2SPROC __glewWindowPos2s = NULL;
PFNGLWINDOWPOS2SVPROC __glewWindowPos2sv = NULL;
PFNGLWINDOWPOS3DPROC __glewWindowPos3d = NULL;
PFNGLWINDOWPOS3DVPROC __glewWindowPos3dv = NULL;
PFNGLWINDOWPOS3FPROC __glewWindowPos3f = NULL;
PFNGLWINDOWPOS3FVPROC __glewWindowPos3fv = NULL;
PFNGLWINDOWPOS3IPROC __glewWindowPos3i = NULL;
PFNGLWINDOWPOS3IVPROC __glewWindowPos3iv = NULL;
PFNGLWINDOWPOS3SPROC __glewWindowPos3s = NULL;
PFNGLWINDOWPOS3SVPROC __glewWindowPos3sv = NULL;

PFNGLBEGINQUERYPROC __glewBeginQuery = NULL;
PFNGLBINDBUFFERPROC __glewBindBuffer = NULL;
PFNGLBUFFERDATAPROC __glewBufferData = NULL;
PFNGLBUFFERSUBDATAPROC __glewBufferSubData = NULL;
PFNGLDELETEBUFFERSPROC __glewDeleteBuffers = NULL;
PFNGLDELETEQUERIESPROC __glewDeleteQueries = NULL;
PFNGLENDQUERYPROC __glewEndQuery = NULL;
PFNGLGENBUFFERSPROC __glewGenBuffers = NULL;
PFNGLGENQUERIESPROC __glewGenQueries = NULL;
PFNGLGETBUFFERPARAMETERIVPROC __glewGetBufferParameteriv = NULL;
PFNGLGETBUFFERPOINTERVPROC __glewGetBufferPointerv = NULL;
PFNGLGETBUFFERSUBDATAPROC __glewGetBufferSubData = NULL;
PFNGLGETQUERYOBJECTIVPROC __glewGetQueryObjectiv = NULL;
PFNGLGETQUERYOBJECTUIVPROC __glewGetQueryObjectuiv = NULL;
PFNGLGETQUERYIVPROC __glewGetQueryiv = NULL;
PFNGLISBUFFERPROC __glewIsBuffer = NULL;
PFNGLISQUERYPROC __glewIsQuery = NULL;
PFNGLMAPBUFFERPROC __glewMapBuffer = NULL;
PFNGLUNMAPBUFFERPROC __glewUnmapBuffer = NULL;

PFNGLATTACHSHADERPROC __glewAttachShader = NULL;
PFNGLBINDATTRIBLOCATIONPROC __glewBindAttribLocation = NULL;
PFNGLBLENDEQUATIONSEPARATEPROC __glewBlendEquationSeparate = NULL;
PFNGLCOMPILESHADERPROC __glewCompileShader = NULL;
PFNGLCREATEPROGRAMPROC __glewCreateProgram = NULL;
PFNGLCREATESHADERPROC __glewCreateShader = NULL;
PFNGLDELETEPROGRAMPROC __glewDeleteProgram = NULL;
PFNGLDELETESHADERPROC __glewDeleteShader = NULL;
PFNGLDETACHSHADERPROC __glewDetachShader = NULL;
PFNGLDISABLEVERTEXATTRIBARRAYPROC __glewDisableVertexAttribArray = NULL;
PFNGLDRAWBUFFERSPROC __glewDrawBuffers = NULL;
PFNGLENABLEVERTEXATTRIBARRAYPROC __glewEnableVertexAttribArray = NULL;
PFNGLGETACTIVEATTRIBPROC __glewGetActiveAttrib = NULL;
PFNGLGETACTIVEUNIFORMPROC __glewGetActiveUniform = NULL;
PFNGLGETATTACHEDSHADERSPROC __glewGetAttachedShaders = NULL;
PFNGLGETATTRIBLOCATIONPROC __glewGetAttribLocation = NULL;
PFNGLGETPROGRAMINFOLOGPROC __glewGetProgramInfoLog = NULL;
PFNGLGETPROGRAMIVPROC __glewGetProgramiv = NULL;
PFNGLGETSHADERINFOLOGPROC __glewGetShaderInfoLog = NULL;
PFNGLGETSHADERSOURCEPROC __glewGetShaderSource = NULL;
PFNGLGETSHADERIVPROC __glewGetShaderiv = NULL;
PFNGLGETUNIFORMLOCATIONPROC __glewGetUniformLocation = NULL;
PFNGLGETUNIFORMFVPROC __glewGetUniformfv = NULL;
PFNGLGETUNIFORMIVPROC __glewGetUniformiv = NULL;
PFNGLGETVERTEXATTRIBPOINTERVPROC __glewGetVertexAttribPointerv = NULL;
PFNGLGETVERTEXATTRIBDVPROC __glewGetVertexAttribdv = NULL;
PFNGLGETVERTEXATTRIBFVPROC __glewGetVertexAttribfv = NULL;
PFNGLGETVERTEXATTRIBIVPROC __glewGetVertexAttribiv = NULL;
PFNGLISPROGRAMPROC __glewIsProgram = NULL;
PFNGLISSHADERPROC __glewIsShader = NULL;
PFNGLLINKPROGRAMPROC __glewLinkProgram = NULL;
PFNGLSHADERSOURCEPROC __glewShaderSource = NULL;
PFNGLSTENCILFUNCSEPARATEPROC __glewStencilFuncSeparate = NULL;
PFNGLSTENCILMASKSEPARATEPROC __glewStencilMaskSeparate = NULL;
PFNGLSTENCILOPSEPARATEPROC __glewStencilOpSeparate = NULL;
PFNGLUNIFORM1FPROC __glewUniform1f = NULL;
PFNGLUNIFORM1FVPROC __glewUniform1fv = NULL;
PFNGLUNIFORM1IPROC __glewUniform1i = NULL;
PFNGLUNIFORM1IVPROC __glewUniform1iv = NULL;
PFNGLUNIFORM2FPROC __glewUniform2f = NULL;
PFNGLUNIFORM2FVPROC __glewUniform2fv = NULL;
PFNGLUNIFORM2IPROC __glewUniform2i = NULL;
PFNGLUNIFORM2IVPROC __glewUniform2iv = NULL;
PFNGLUNIFORM3FPROC __glewUniform3f = NULL;
PFNGLUNIFORM3FVPROC __glewUniform3fv = NULL;
PFNGLUNIFORM3IPROC __glewUniform3i = NULL;
PFNGLUNIFORM3IVPROC __glewUniform3iv = NULL;
PFNGLUNIFORM4FPROC __glewUniform4f = NULL;
PFNGLUNIFORM4FVPROC __glewUniform4fv = NULL;
PFNGLUNIFORM4IPROC __glewUniform4i = NULL;
PFNGLUNIFORM4IVPROC __glewUniform4iv = NULL;
PFNGLUNIFORMMATRIX2FVPROC __glewUniformMatrix2fv = NULL;
PFNGLUNIFORMMATRIX3FVPROC __glewUniformMatrix3fv = NULL;
PFNGLUNIFORMMATRIX4FVPROC __glewUniformMatrix4fv = NULL;
PFNGLUSEPROGRAMPROC __glewUseProgram = NULL;
PFNGLVALIDATEPROGRAMPROC __glewValidateProgram = NULL;
PFNGLVERTEXATTRIB1DPROC __glewVertexAttrib1d = NULL;
PFNGLVERTEXATTRIB1DVPROC __glewVertexAttrib1dv = NULL;
PFNGLVERTEXATTRIB1FPROC __glewVertexAttrib1f = NULL;
PFNGLVERTEXATTRIB1FVPROC __glewVertexAttrib1fv = NULL;
PFNGLVERTEXATTRIB1SPROC __glewVertexAttrib1s = NULL;
PFNGLVERTEXATTRIB1SVPROC __glewVertexAttrib1sv = NULL;
PFNGLVERTEXATTRIB2DPROC __glewVertexAttrib2d = NULL;
PFNGLVERTEXATTRIB2DVPROC __glewVertexAttrib2dv = NULL;
PFNGLVERTEXATTRIB2FPROC __glewVertexAttrib2f = NULL;
PFNGLVERTEXATTRIB2FVPROC __glewVertexAttrib2fv = NULL;
PFNGLVERTEXATTRIB2SPROC __glewVertexAttrib2s = NULL;
PFNGLVERTEXATTRIB2SVPROC __glewVertexAttrib2sv = NULL;
PFNGLVERTEXATTRIB3DPROC __glewVertexAttrib3d = NULL;
PFNGLVERTEXATTRIB3DVPROC __glewVertexAttrib3dv = NULL;
PFNGLVERTEXATTRIB3FPROC __glewVertexAttrib3f = NULL;
PFNGLVERTEXATTRIB3FVPROC __glewVertexAttrib3fv = NULL;
PFNGLVERTEXATTRIB3SPROC __glewVertexAttrib3s = NULL;
PFNGLVERTEXATTRIB3SVPROC __glewVertexAttrib3sv = NULL;
PFNGLVERTEXATTRIB4NBVPROC __glewVertexAttrib4Nbv = NULL;
PFNGLVERTEXATTRIB4NIVPROC __glewVertexAttrib4Niv = NULL;
PFNGLVERTEXATTRIB4NSVPROC __glewVertexAttrib4Nsv = NULL;
PFNGLVERTEXATTRIB4NUBPROC __glewVertexAttrib4Nub = NULL;
PFNGLVERTEXATTRIB4NUBVPROC __glewVertexAttrib4Nubv = NULL;
PFNGLVERTEXATTRIB4NUIVPROC __glewVertexAttrib4Nuiv = NULL;
PFNGLVERTEXATTRIB4NUSVPROC __glewVertexAttrib4Nusv = NULL;
PFNGLVERTEXATTRIB4BVPROC __glewVertexAttrib4bv = NULL;
PFNGLVERTEXATTRIB4DPROC __glewVertexAttrib4d = NULL;
PFNGLVERTEXATTRIB4DVPROC __glewVertexAttrib4dv = NULL;
PFNGLVERTEXATTRIB4FPROC __glewVertexAttrib4f = NULL;
PFNGLVERTEXATTRIB4FVPROC __glewVertexAttrib4fv = NULL;
PFNGLVERTEXATTRIB4IVPROC __glewVertexAttrib4iv = NULL;
PFNGLVERTEXATTRIB4SPROC __glewVertexAttrib4s = NULL;
PFNGLVERTEXATTRIB4SVPROC __glewVertexAttrib4sv = NULL;
PFNGLVERTEXATTRIB4UBVPROC __glewVertexAttrib4ubv = NULL;
PFNGLVERTEXATTRIB4UIVPROC __glewVertexAttrib4uiv = NULL;
PFNGLVERTEXATTRIB4USVPROC __glewVertexAttrib4usv = NULL;
PFNGLVERTEXATTRIBPOINTERPROC __glewVertexAttribPointer = NULL;

PFNGLUNIFORMMATRIX2X3FVPROC __glewUniformMatrix2x3fv = NULL;
PFNGLUNIFORMMATRIX2X4FVPROC __glewUniformMatrix2x4fv = NULL;
PFNGLUNIFORMMATRIX3X2FVPROC __glewUniformMatrix3x2fv = NULL;
PFNGLUNIFORMMATRIX3X4FVPROC __glewUniformMatrix3x4fv = NULL;
PFNGLUNIFORMMATRIX4X2FVPROC __glewUniformMatrix4x2fv = NULL;
PFNGLUNIFORMMATRIX4X3FVPROC __glewUniformMatrix4x3fv = NULL;

PFNGLBEGINCONDITIONALRENDERPROC __glewBeginConditionalRender = NULL;
PFNGLBEGINTRANSFORMFEEDBACKPROC __glewBeginTransformFeedback = NULL;
PFNGLBINDFRAGDATALOCATIONPROC __glewBindFragDataLocation = NULL;
PFNGLCLAMPCOLORPROC __glewClampColor = NULL;
PFNGLCLEARBUFFERFIPROC __glewClearBufferfi = NULL;
PFNGLCLEARBUFFERFVPROC __glewClearBufferfv = NULL;
PFNGLCLEARBUFFERIVPROC __glewClearBufferiv = NULL;
PFNGLCLEARBUFFERUIVPROC __glewClearBufferuiv = NULL;
PFNGLCOLORMASKIPROC __glewColorMaski = NULL;
PFNGLDISABLEIPROC __glewDisablei = NULL;
PFNGLENABLEIPROC __glewEnablei = NULL;
PFNGLENDCONDITIONALRENDERPROC __glewEndConditionalRender = NULL;
PFNGLENDTRANSFORMFEEDBACKPROC __glewEndTransformFeedback = NULL;
PFNGLGETBOOLEANI_VPROC __glewGetBooleani_v = NULL;
PFNGLGETFRAGDATALOCATIONPROC __glewGetFragDataLocation = NULL;
PFNGLGETSTRINGIPROC __glewGetStringi = NULL;
PFNGLGETTEXPARAMETERIIVPROC __glewGetTexParameterIiv = NULL;
PFNGLGETTEXPARAMETERIUIVPROC __glewGetTexParameterIuiv = NULL;
PFNGLGETTRANSFORMFEEDBACKVARYINGPROC __glewGetTransformFeedbackVarying = NULL;
PFNGLGETUNIFORMUIVPROC __glewGetUniformuiv = NULL;
PFNGLGETVERTEXATTRIBIIVPROC __glewGetVertexAttribIiv = NULL;
PFNGLGETVERTEXATTRIBIUIVPROC __glewGetVertexAttribIuiv = NULL;
PFNGLISENABLEDIPROC __glewIsEnabledi = NULL;
PFNGLTEXPARAMETERIIVPROC __glewTexParameterIiv = NULL;
PFNGLTEXPARAMETERIUIVPROC __glewTexParameterIuiv = NULL;
PFNGLTRANSFORMFEEDBACKVARYINGSPROC __glewTransformFeedbackVaryings = NULL;
PFNGLUNIFORM1UIPROC __glewUniform1ui = NULL;
PFNGLUNIFORM1UIVPROC __glewUniform1uiv = NULL;
PFNGLUNIFORM2UIPROC __glewUniform2ui = NULL;
PFNGLUNIFORM2UIVPROC __glewUniform2uiv = NULL;
PFNGLUNIFORM3UIPROC __glewUniform3ui = NULL;
PFNGLUNIFORM3UIVPROC __glewUniform3uiv = NULL;
PFNGLUNIFORM4UIPROC __glewUniform4ui = NULL;
PFNGLUNIFORM4UIVPROC __glewUniform4uiv = NULL;
PFNGLVERTEXATTRIBI1IPROC __glewVertexAttribI1i = NULL;
PFNGLVERTEXATTRIBI1IVPROC __glewVertexAttribI1iv = NULL;
PFNGLVERTEXATTRIBI1UIPROC __glewVertexAttribI1ui = NULL;
PFNGLVERTEXATTRIBI1UIVPROC __glewVertexAttribI1uiv = NULL;
PFNGLVERTEXATTRIBI2IPROC __glewVertexAttribI2i = NULL;
PFNGLVERTEXATTRIBI2IVPROC __glewVertexAttribI2iv = NULL;
PFNGLVERTEXATTRIBI2UIPROC __glewVertexAttribI2ui = NULL;
PFNGLVERTEXATTRIBI2UIVPROC __glewVertexAttribI2uiv = NULL;
PFNGLVERTEXATTRIBI3IPROC __glewVertexAttribI3i = NULL;
PFNGLVERTEXATTRIBI3IVPROC __glewVertexAttribI3iv = NULL;
PFNGLVERTEXATTRIBI3UIPROC __glewVertexAttribI3ui = NULL;
PFNGLVERTEXATTRIBI3UIVPROC __glewVertexAttribI3uiv = NULL;
PFNGLVERTEXATTRIBI4BVPROC __glewVertexAttribI4bv = NULL;
PFNGLVERTEXATTRIBI4IPROC __glewVertexAttribI4i = NULL;
PFNGLVERTEXATTRIBI4IVPROC __glewVertexAttribI4iv = NULL;
PFNGLVERTEXATTRIBI4SVPROC __glewVertexAttribI4sv = NULL;
PFNGLVERTEXATTRIBI4UBVPROC __glewVertexAttribI4ubv = NULL;
PFNGLVERTEXATTRIBI4UIPROC __glewVertexAttribI4ui = NULL;
PFNGLVERTEXATTRIBI4UIVPROC __glewVertexAttribI4uiv = NULL;
PFNGLVERTEXATTRIBI4USVPROC __glewVertexAttribI4usv = NULL;
PFNGLVERTEXATTRIBIPOINTERPROC __glewVertexAttribIPointer = NULL;

PFNGLDRAWARRAYSINSTANCEDPROC __glewDrawArraysInstanced = NULL;
PFNGLDRAWELEMENTSINSTANCEDPROC __glewDrawElementsInstanced = NULL;
PFNGLPRIMITIVERESTARTINDEXPROC __glewPrimitiveRestartIndex = NULL;
PFNGLTEXBUFFERPROC __glewTexBuffer = NULL;

PFNGLFRAMEBUFFERTEXTUREPROC __glewFramebufferTexture = NULL;
PFNGLGETBUFFERPARAMETERI64VPROC __glewGetBufferParameteri64v = NULL;
PFNGLGETINTEGER64I_VPROC __glewGetInteger64i_v = NULL;

PFNGLVERTEXATTRIBDIVISORPROC __glewVertexAttribDivisor = NULL;

PFNGLBLENDEQUATIONSEPARATEIPROC __glewBlendEquationSeparatei = NULL;
PFNGLBLENDEQUATIONIPROC __glewBlendEquationi = NULL;
PFNGLBLENDFUNCSEPARATEIPROC __glewBlendFuncSeparatei = NULL;
PFNGLBLENDFUNCIPROC __glewBlendFunci = NULL;
PFNGLMINSAMPLESHADINGPROC __glewMinSampleShading = NULL;

PFNGLGETGRAPHICSRESETSTATUSPROC __glewGetGraphicsResetStatus = NULL;
PFNGLGETNCOMPRESSEDTEXIMAGEPROC __glewGetnCompressedTexImage = NULL;
PFNGLGETNTEXIMAGEPROC __glewGetnTexImage = NULL;
PFNGLGETNUNIFORMDVPROC __glewGetnUniformdv = NULL;

PFNGLTBUFFERMASK3DFXPROC __glewTbufferMask3DFX = NULL;

PFNGLDEBUGMESSAGECALLBACKAMDPROC __glewDebugMessageCallbackAMD = NULL;
PFNGLDEBUGMESSAGEENABLEAMDPROC __glewDebugMessageEnableAMD = NULL;
PFNGLDEBUGMESSAGEINSERTAMDPROC __glewDebugMessageInsertAMD = NULL;
PFNGLGETDEBUGMESSAGELOGAMDPROC __glewGetDebugMessageLogAMD = NULL;

PFNGLBLENDEQUATIONINDEXEDAMDPROC __glewBlendEquationIndexedAMD = NULL;
PFNGLBLENDEQUATIONSEPARATEINDEXEDAMDPROC __glewBlendEquationSeparateIndexedAMD = NULL;
PFNGLBLENDFUNCINDEXEDAMDPROC __glewBlendFuncIndexedAMD = NULL;
PFNGLBLENDFUNCSEPARATEINDEXEDAMDPROC __glewBlendFuncSeparateIndexedAMD = NULL;

PFNGLVERTEXATTRIBPARAMETERIAMDPROC __glewVertexAttribParameteriAMD = NULL;

PFNGLMULTIDRAWARRAYSINDIRECTAMDPROC __glewMultiDrawArraysIndirectAMD = NULL;
PFNGLMULTIDRAWELEMENTSINDIRECTAMDPROC __glewMultiDrawElementsIndirectAMD = NULL;

PFNGLDELETENAMESAMDPROC __glewDeleteNamesAMD = NULL;
PFNGLGENNAMESAMDPROC __glewGenNamesAMD = NULL;
PFNGLISNAMEAMDPROC __glewIsNameAMD = NULL;

PFNGLQUERYOBJECTPARAMETERUIAMDPROC __glewQueryObjectParameteruiAMD = NULL;

PFNGLBEGINPERFMONITORAMDPROC __glewBeginPerfMonitorAMD = NULL;
PFNGLDELETEPERFMONITORSAMDPROC __glewDeletePerfMonitorsAMD = NULL;
PFNGLENDPERFMONITORAMDPROC __glewEndPerfMonitorAMD = NULL;
PFNGLGENPERFMONITORSAMDPROC __glewGenPerfMonitorsAMD = NULL;
PFNGLGETPERFMONITORCOUNTERDATAAMDPROC __glewGetPerfMonitorCounterDataAMD = NULL;
PFNGLGETPERFMONITORCOUNTERINFOAMDPROC __glewGetPerfMonitorCounterInfoAMD = NULL;
PFNGLGETPERFMONITORCOUNTERSTRINGAMDPROC __glewGetPerfMonitorCounterStringAMD = NULL;
PFNGLGETPERFMONITORCOUNTERSAMDPROC __glewGetPerfMonitorCountersAMD = NULL;
PFNGLGETPERFMONITORGROUPSTRINGAMDPROC __glewGetPerfMonitorGroupStringAMD = NULL;
PFNGLGETPERFMONITORGROUPSAMDPROC __glewGetPerfMonitorGroupsAMD = NULL;
PFNGLSELECTPERFMONITORCOUNTERSAMDPROC __glewSelectPerfMonitorCountersAMD = NULL;

PFNGLSETMULTISAMPLEFVAMDPROC __glewSetMultisamplefvAMD = NULL;

PFNGLTEXSTORAGESPARSEAMDPROC __glewTexStorageSparseAMD = NULL;
PFNGLTEXTURESTORAGESPARSEAMDPROC __glewTextureStorageSparseAMD = NULL;

PFNGLSTENCILOPVALUEAMDPROC __glewStencilOpValueAMD = NULL;

PFNGLTESSELLATIONFACTORAMDPROC __glewTessellationFactorAMD = NULL;
PFNGLTESSELLATIONMODEAMDPROC __glewTessellationModeAMD = NULL;

PFNGLBLITFRAMEBUFFERANGLEPROC __glewBlitFramebufferANGLE = NULL;

PFNGLRENDERBUFFERSTORAGEMULTISAMPLEANGLEPROC __glewRenderbufferStorageMultisampleANGLE = NULL;

PFNGLDRAWARRAYSINSTANCEDANGLEPROC __glewDrawArraysInstancedANGLE = NULL;
PFNGLDRAWELEMENTSINSTANCEDANGLEPROC __glewDrawElementsInstancedANGLE = NULL;
PFNGLVERTEXATTRIBDIVISORANGLEPROC __glewVertexAttribDivisorANGLE = NULL;

PFNGLBEGINQUERYANGLEPROC __glewBeginQueryANGLE = NULL;
PFNGLDELETEQUERIESANGLEPROC __glewDeleteQueriesANGLE = NULL;
PFNGLENDQUERYANGLEPROC __glewEndQueryANGLE = NULL;
PFNGLGENQUERIESANGLEPROC __glewGenQueriesANGLE = NULL;
PFNGLGETQUERYOBJECTI64VANGLEPROC __glewGetQueryObjecti64vANGLE = NULL;
PFNGLGETQUERYOBJECTIVANGLEPROC __glewGetQueryObjectivANGLE = NULL;
PFNGLGETQUERYOBJECTUI64VANGLEPROC __glewGetQueryObjectui64vANGLE = NULL;
PFNGLGETQUERYOBJECTUIVANGLEPROC __glewGetQueryObjectuivANGLE = NULL;
PFNGLGETQUERYIVANGLEPROC __glewGetQueryivANGLE = NULL;
PFNGLISQUERYANGLEPROC __glewIsQueryANGLE = NULL;
PFNGLQUERYCOUNTERANGLEPROC __glewQueryCounterANGLE = NULL;

PFNGLGETTRANSLATEDSHADERSOURCEANGLEPROC __glewGetTranslatedShaderSourceANGLE = NULL;

PFNGLDRAWELEMENTARRAYAPPLEPROC __glewDrawElementArrayAPPLE = NULL;
PFNGLDRAWRANGEELEMENTARRAYAPPLEPROC __glewDrawRangeElementArrayAPPLE = NULL;
PFNGLELEMENTPOINTERAPPLEPROC __glewElementPointerAPPLE = NULL;
PFNGLMULTIDRAWELEMENTARRAYAPPLEPROC __glewMultiDrawElementArrayAPPLE = NULL;
PFNGLMULTIDRAWRANGEELEMENTARRAYAPPLEPROC __glewMultiDrawRangeElementArrayAPPLE = NULL;

PFNGLDELETEFENCESAPPLEPROC __glewDeleteFencesAPPLE = NULL;
PFNGLFINISHFENCEAPPLEPROC __glewFinishFenceAPPLE = NULL;
PFNGLFINISHOBJECTAPPLEPROC __glewFinishObjectAPPLE = NULL;
PFNGLGENFENCESAPPLEPROC __glewGenFencesAPPLE = NULL;
PFNGLISFENCEAPPLEPROC __glewIsFenceAPPLE = NULL;
PFNGLSETFENCEAPPLEPROC __glewSetFenceAPPLE = NULL;
PFNGLTESTFENCEAPPLEPROC __glewTestFenceAPPLE = NULL;
PFNGLTESTOBJECTAPPLEPROC __glewTestObjectAPPLE = NULL;

PFNGLBUFFERPARAMETERIAPPLEPROC __glewBufferParameteriAPPLE = NULL;
PFNGLFLUSHMAPPEDBUFFERRANGEAPPLEPROC __glewFlushMappedBufferRangeAPPLE = NULL;

PFNGLGETOBJECTPARAMETERIVAPPLEPROC __glewGetObjectParameterivAPPLE = NULL;
PFNGLOBJECTPURGEABLEAPPLEPROC __glewObjectPurgeableAPPLE = NULL;
PFNGLOBJECTUNPURGEABLEAPPLEPROC __glewObjectUnpurgeableAPPLE = NULL;

PFNGLGETTEXPARAMETERPOINTERVAPPLEPROC __glewGetTexParameterPointervAPPLE = NULL;
PFNGLTEXTURERANGEAPPLEPROC __glewTextureRangeAPPLE = NULL;

PFNGLBINDVERTEXARRAYAPPLEPROC __glewBindVertexArrayAPPLE = NULL;
PFNGLDELETEVERTEXARRAYSAPPLEPROC __glewDeleteVertexArraysAPPLE = NULL;
PFNGLGENVERTEXARRAYSAPPLEPROC __glewGenVertexArraysAPPLE = NULL;
PFNGLISVERTEXARRAYAPPLEPROC __glewIsVertexArrayAPPLE = NULL;

PFNGLFLUSHVERTEXARRAYRANGEAPPLEPROC __glewFlushVertexArrayRangeAPPLE = NULL;
PFNGLVERTEXARRAYPARAMETERIAPPLEPROC __glewVertexArrayParameteriAPPLE = NULL;
PFNGLVERTEXARRAYRANGEAPPLEPROC __glewVertexArrayRangeAPPLE = NULL;

PFNGLDISABLEVERTEXATTRIBAPPLEPROC __glewDisableVertexAttribAPPLE = NULL;
PFNGLENABLEVERTEXATTRIBAPPLEPROC __glewEnableVertexAttribAPPLE = NULL;
PFNGLISVERTEXATTRIBENABLEDAPPLEPROC __glewIsVertexAttribEnabledAPPLE = NULL;
PFNGLMAPVERTEXATTRIB1DAPPLEPROC __glewMapVertexAttrib1dAPPLE = NULL;
PFNGLMAPVERTEXATTRIB1FAPPLEPROC __glewMapVertexAttrib1fAPPLE = NULL;
PFNGLMAPVERTEXATTRIB2DAPPLEPROC __glewMapVertexAttrib2dAPPLE = NULL;
PFNGLMAPVERTEXATTRIB2FAPPLEPROC __glewMapVertexAttrib2fAPPLE = NULL;

PFNGLCLEARDEPTHFPROC __glewClearDepthf = NULL;
PFNGLDEPTHRANGEFPROC __glewDepthRangef = NULL;
PFNGLGETSHADERPRECISIONFORMATPROC __glewGetShaderPrecisionFormat = NULL;
PFNGLRELEASESHADERCOMPILERPROC __glewReleaseShaderCompiler = NULL;
PFNGLSHADERBINARYPROC __glewShaderBinary = NULL;

PFNGLMEMORYBARRIERBYREGIONPROC __glewMemoryBarrierByRegion = NULL;

PFNGLPRIMITIVEBOUNDINGBOXARBPROC __glewPrimitiveBoundingBoxARB = NULL;

PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC __glewDrawArraysInstancedBaseInstance = NULL;
PFNGLDRAWELEMENTSINSTANCEDBASEINSTANCEPROC __glewDrawElementsInstancedBaseInstance = NULL;
PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC __glewDrawElementsInstancedBaseVertexBaseInstance = NULL;

PFNGLGETIMAGEHANDLEARBPROC __glewGetImageHandleARB = NULL;
PFNGLGETTEXTUREHANDLEARBPROC __glewGetTextureHandleARB = NULL;
PFNGLGETTEXTURESAMPLERHANDLEARBPROC __glewGetTextureSamplerHandleARB = NULL;
PFNGLGETVERTEXATTRIBLUI64VARBPROC __glewGetVertexAttribLui64vARB = NULL;
PFNGLISIMAGEHANDLERESIDENTARBPROC __glewIsImageHandleResidentARB = NULL;
PFNGLISTEXTUREHANDLERESIDENTARBPROC __glewIsTextureHandleResidentARB = NULL;
PFNGLMAKEIMAGEHANDLENONRESIDENTARBPROC __glewMakeImageHandleNonResidentARB = NULL;
PFNGLMAKEIMAGEHANDLERESIDENTARBPROC __glewMakeImageHandleResidentARB = NULL;
PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC __glewMakeTextureHandleNonResidentARB = NULL;
PFNGLMAKETEXTUREHANDLERESIDENTARBPROC __glewMakeTextureHandleResidentARB = NULL;
PFNGLPROGRAMUNIFORMHANDLEUI64ARBPROC __glewProgramUniformHandleui64ARB = NULL;
PFNGLPROGRAMUNIFORMHANDLEUI64VARBPROC __glewProgramUniformHandleui64vARB = NULL;
PFNGLUNIFORMHANDLEUI64ARBPROC __glewUniformHandleui64ARB = NULL;
PFNGLUNIFORMHANDLEUI64VARBPROC __glewUniformHandleui64vARB = NULL;
PFNGLVERTEXATTRIBL1UI64ARBPROC __glewVertexAttribL1ui64ARB = NULL;
PFNGLVERTEXATTRIBL1UI64VARBPROC __glewVertexAttribL1ui64vARB = NULL;

PFNGLBINDFRAGDATALOCATIONINDEXEDPROC __glewBindFragDataLocationIndexed = NULL;
PFNGLGETFRAGDATAINDEXPROC __glewGetFragDataIndex = NULL;

PFNGLBUFFERSTORAGEPROC __glewBufferStorage = NULL;
PFNGLNAMEDBUFFERSTORAGEEXTPROC __glewNamedBufferStorageEXT = NULL;

PFNGLCREATESYNCFROMCLEVENTARBPROC __glewCreateSyncFromCLeventARB = NULL;

PFNGLCLEARBUFFERDATAPROC __glewClearBufferData = NULL;
PFNGLCLEARBUFFERSUBDATAPROC __glewClearBufferSubData = NULL;
PFNGLCLEARNAMEDBUFFERDATAEXTPROC __glewClearNamedBufferDataEXT = NULL;
PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC __glewClearNamedBufferSubDataEXT = NULL;

PFNGLCLEARTEXIMAGEPROC __glewClearTexImage = NULL;
PFNGLCLEARTEXSUBIMAGEPROC __glewClearTexSubImage = NULL;

PFNGLCLIPCONTROLPROC __glewClipControl = NULL;

PFNGLCLAMPCOLORARBPROC __glewClampColorARB = NULL;

PFNGLDISPATCHCOMPUTEPROC __glewDispatchCompute = NULL;
PFNGLDISPATCHCOMPUTEINDIRECTPROC __glewDispatchComputeIndirect = NULL;

PFNGLDISPATCHCOMPUTEGROUPSIZEARBPROC __glewDispatchComputeGroupSizeARB = NULL;

PFNGLCOPYBUFFERSUBDATAPROC __glewCopyBufferSubData = NULL;

PFNGLCOPYIMAGESUBDATAPROC __glewCopyImageSubData = NULL;

PFNGLDEBUGMESSAGECALLBACKARBPROC __glewDebugMessageCallbackARB = NULL;
PFNGLDEBUGMESSAGECONTROLARBPROC __glewDebugMessageControlARB = NULL;
PFNGLDEBUGMESSAGEINSERTARBPROC __glewDebugMessageInsertARB = NULL;
PFNGLGETDEBUGMESSAGELOGARBPROC __glewGetDebugMessageLogARB = NULL;

PFNGLBINDTEXTUREUNITPROC __glewBindTextureUnit = NULL;
PFNGLBLITNAMEDFRAMEBUFFERPROC __glewBlitNamedFramebuffer = NULL;
PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC __glewCheckNamedFramebufferStatus = NULL;
PFNGLCLEARNAMEDBUFFERDATAPROC __glewClearNamedBufferData = NULL;
PFNGLCLEARNAMEDBUFFERSUBDATAPROC __glewClearNamedBufferSubData = NULL;
PFNGLCLEARNAMEDFRAMEBUFFERFIPROC __glewClearNamedFramebufferfi = NULL;
PFNGLCLEARNAMEDFRAMEBUFFERFVPROC __glewClearNamedFramebufferfv = NULL;
PFNGLCLEARNAMEDFRAMEBUFFERIVPROC __glewClearNamedFramebufferiv = NULL;
PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC __glewClearNamedFramebufferuiv = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC __glewCompressedTextureSubImage1D = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC __glewCompressedTextureSubImage2D = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC __glewCompressedTextureSubImage3D = NULL;
PFNGLCOPYNAMEDBUFFERSUBDATAPROC __glewCopyNamedBufferSubData = NULL;
PFNGLCOPYTEXTURESUBIMAGE1DPROC __glewCopyTextureSubImage1D = NULL;
PFNGLCOPYTEXTURESUBIMAGE2DPROC __glewCopyTextureSubImage2D = NULL;
PFNGLCOPYTEXTURESUBIMAGE3DPROC __glewCopyTextureSubImage3D = NULL;
PFNGLCREATEBUFFERSPROC __glewCreateBuffers = NULL;
PFNGLCREATEFRAMEBUFFERSPROC __glewCreateFramebuffers = NULL;
PFNGLCREATEPROGRAMPIPELINESPROC __glewCreateProgramPipelines = NULL;
PFNGLCREATEQUERIESPROC __glewCreateQueries = NULL;
PFNGLCREATERENDERBUFFERSPROC __glewCreateRenderbuffers = NULL;
PFNGLCREATESAMPLERSPROC __glewCreateSamplers = NULL;
PFNGLCREATETEXTURESPROC __glewCreateTextures = NULL;
PFNGLCREATETRANSFORMFEEDBACKSPROC __glewCreateTransformFeedbacks = NULL;
PFNGLCREATEVERTEXARRAYSPROC __glewCreateVertexArrays = NULL;
PFNGLDISABLEVERTEXARRAYATTRIBPROC __glewDisableVertexArrayAttrib = NULL;
PFNGLENABLEVERTEXARRAYATTRIBPROC __glewEnableVertexArrayAttrib = NULL;
PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC __glewFlushMappedNamedBufferRange = NULL;
PFNGLGENERATETEXTUREMIPMAPPROC __glewGenerateTextureMipmap = NULL;
PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC __glewGetCompressedTextureImage = NULL;
PFNGLGETNAMEDBUFFERPARAMETERI64VPROC __glewGetNamedBufferParameteri64v = NULL;
PFNGLGETNAMEDBUFFERPARAMETERIVPROC __glewGetNamedBufferParameteriv = NULL;
PFNGLGETNAMEDBUFFERPOINTERVPROC __glewGetNamedBufferPointerv = NULL;
PFNGLGETNAMEDBUFFERSUBDATAPROC __glewGetNamedBufferSubData = NULL;
PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVPROC __glewGetNamedFramebufferAttachmentParameteriv = NULL;
PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVPROC __glewGetNamedFramebufferParameteriv = NULL;
PFNGLGETNAMEDRENDERBUFFERPARAMETERIVPROC __glewGetNamedRenderbufferParameteriv = NULL;
PFNGLGETQUERYBUFFEROBJECTI64VPROC __glewGetQueryBufferObjecti64v = NULL;
PFNGLGETQUERYBUFFEROBJECTIVPROC __glewGetQueryBufferObjectiv = NULL;
PFNGLGETQUERYBUFFEROBJECTUI64VPROC __glewGetQueryBufferObjectui64v = NULL;
PFNGLGETQUERYBUFFEROBJECTUIVPROC __glewGetQueryBufferObjectuiv = NULL;
PFNGLGETTEXTUREIMAGEPROC __glewGetTextureImage = NULL;
PFNGLGETTEXTURELEVELPARAMETERFVPROC __glewGetTextureLevelParameterfv = NULL;
PFNGLGETTEXTURELEVELPARAMETERIVPROC __glewGetTextureLevelParameteriv = NULL;
PFNGLGETTEXTUREPARAMETERIIVPROC __glewGetTextureParameterIiv = NULL;
PFNGLGETTEXTUREPARAMETERIUIVPROC __glewGetTextureParameterIuiv = NULL;
PFNGLGETTEXTUREPARAMETERFVPROC __glewGetTextureParameterfv = NULL;
PFNGLGETTEXTUREPARAMETERIVPROC __glewGetTextureParameteriv = NULL;
PFNGLGETTRANSFORMFEEDBACKI64_VPROC __glewGetTransformFeedbacki64_v = NULL;
PFNGLGETTRANSFORMFEEDBACKI_VPROC __glewGetTransformFeedbacki_v = NULL;
PFNGLGETTRANSFORMFEEDBACKIVPROC __glewGetTransformFeedbackiv = NULL;
PFNGLGETVERTEXARRAYINDEXED64IVPROC __glewGetVertexArrayIndexed64iv = NULL;
PFNGLGETVERTEXARRAYINDEXEDIVPROC __glewGetVertexArrayIndexediv = NULL;
PFNGLGETVERTEXARRAYIVPROC __glewGetVertexArrayiv = NULL;
PFNGLINVALIDATENAMEDFRAMEBUFFERDATAPROC __glewInvalidateNamedFramebufferData = NULL;
PFNGLINVALIDATENAMEDFRAMEBUFFERSUBDATAPROC __glewInvalidateNamedFramebufferSubData = NULL;
PFNGLMAPNAMEDBUFFERPROC __glewMapNamedBuffer = NULL;
PFNGLMAPNAMEDBUFFERRANGEPROC __glewMapNamedBufferRange = NULL;
PFNGLNAMEDBUFFERDATAPROC __glewNamedBufferData = NULL;
PFNGLNAMEDBUFFERSTORAGEPROC __glewNamedBufferStorage = NULL;
PFNGLNAMEDBUFFERSUBDATAPROC __glewNamedBufferSubData = NULL;
PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC __glewNamedFramebufferDrawBuffer = NULL;
PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC __glewNamedFramebufferDrawBuffers = NULL;
PFNGLNAMEDFRAMEBUFFERPARAMETERIPROC __glewNamedFramebufferParameteri = NULL;
PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC __glewNamedFramebufferReadBuffer = NULL;
PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC __glewNamedFramebufferRenderbuffer = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTUREPROC __glewNamedFramebufferTexture = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC __glewNamedFramebufferTextureLayer = NULL;
PFNGLNAMEDRENDERBUFFERSTORAGEPROC __glewNamedRenderbufferStorage = NULL;
PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEPROC __glewNamedRenderbufferStorageMultisample = NULL;
PFNGLTEXTUREBUFFERPROC __glewTextureBuffer = NULL;
PFNGLTEXTUREBUFFERRANGEPROC __glewTextureBufferRange = NULL;
PFNGLTEXTUREPARAMETERIIVPROC __glewTextureParameterIiv = NULL;
PFNGLTEXTUREPARAMETERIUIVPROC __glewTextureParameterIuiv = NULL;
PFNGLTEXTUREPARAMETERFPROC __glewTextureParameterf = NULL;
PFNGLTEXTUREPARAMETERFVPROC __glewTextureParameterfv = NULL;
PFNGLTEXTUREPARAMETERIPROC __glewTextureParameteri = NULL;
PFNGLTEXTUREPARAMETERIVPROC __glewTextureParameteriv = NULL;
PFNGLTEXTURESTORAGE1DPROC __glewTextureStorage1D = NULL;
PFNGLTEXTURESTORAGE2DPROC __glewTextureStorage2D = NULL;
PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC __glewTextureStorage2DMultisample = NULL;
PFNGLTEXTURESTORAGE3DPROC __glewTextureStorage3D = NULL;
PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC __glewTextureStorage3DMultisample = NULL;
PFNGLTEXTURESUBIMAGE1DPROC __glewTextureSubImage1D = NULL;
PFNGLTEXTURESUBIMAGE2DPROC __glewTextureSubImage2D = NULL;
PFNGLTEXTURESUBIMAGE3DPROC __glewTextureSubImage3D = NULL;
PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC __glewTransformFeedbackBufferBase = NULL;
PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC __glewTransformFeedbackBufferRange = NULL;
PFNGLUNMAPNAMEDBUFFERPROC __glewUnmapNamedBuffer = NULL;
PFNGLVERTEXARRAYATTRIBBINDINGPROC __glewVertexArrayAttribBinding = NULL;
PFNGLVERTEXARRAYATTRIBFORMATPROC __glewVertexArrayAttribFormat = NULL;
PFNGLVERTEXARRAYATTRIBIFORMATPROC __glewVertexArrayAttribIFormat = NULL;
PFNGLVERTEXARRAYATTRIBLFORMATPROC __glewVertexArrayAttribLFormat = NULL;
PFNGLVERTEXARRAYBINDINGDIVISORPROC __glewVertexArrayBindingDivisor = NULL;
PFNGLVERTEXARRAYELEMENTBUFFERPROC __glewVertexArrayElementBuffer = NULL;
PFNGLVERTEXARRAYVERTEXBUFFERPROC __glewVertexArrayVertexBuffer = NULL;
PFNGLVERTEXARRAYVERTEXBUFFERSPROC __glewVertexArrayVertexBuffers = NULL;

PFNGLDRAWBUFFERSARBPROC __glewDrawBuffersARB = NULL;

PFNGLBLENDEQUATIONSEPARATEIARBPROC __glewBlendEquationSeparateiARB = NULL;
PFNGLBLENDEQUATIONIARBPROC __glewBlendEquationiARB = NULL;
PFNGLBLENDFUNCSEPARATEIARBPROC __glewBlendFuncSeparateiARB = NULL;
PFNGLBLENDFUNCIARBPROC __glewBlendFunciARB = NULL;

PFNGLDRAWELEMENTSBASEVERTEXPROC __glewDrawElementsBaseVertex = NULL;
PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC __glewDrawElementsInstancedBaseVertex = NULL;
PFNGLDRAWRANGEELEMENTSBASEVERTEXPROC __glewDrawRangeElementsBaseVertex = NULL;
PFNGLMULTIDRAWELEMENTSBASEVERTEXPROC __glewMultiDrawElementsBaseVertex = NULL;

PFNGLDRAWARRAYSINDIRECTPROC __glewDrawArraysIndirect = NULL;
PFNGLDRAWELEMENTSINDIRECTPROC __glewDrawElementsIndirect = NULL;

PFNGLFRAMEBUFFERPARAMETERIPROC __glewFramebufferParameteri = NULL;
PFNGLGETFRAMEBUFFERPARAMETERIVPROC __glewGetFramebufferParameteriv = NULL;
PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVEXTPROC __glewGetNamedFramebufferParameterivEXT = NULL;
PFNGLNAMEDFRAMEBUFFERPARAMETERIEXTPROC __glewNamedFramebufferParameteriEXT = NULL;

PFNGLBINDFRAMEBUFFERPROC __glewBindFramebuffer = NULL;
PFNGLBINDRENDERBUFFERPROC __glewBindRenderbuffer = NULL;
PFNGLBLITFRAMEBUFFERPROC __glewBlitFramebuffer = NULL;
PFNGLCHECKFRAMEBUFFERSTATUSPROC __glewCheckFramebufferStatus = NULL;
PFNGLDELETEFRAMEBUFFERSPROC __glewDeleteFramebuffers = NULL;
PFNGLDELETERENDERBUFFERSPROC __glewDeleteRenderbuffers = NULL;
PFNGLFRAMEBUFFERRENDERBUFFERPROC __glewFramebufferRenderbuffer = NULL;
PFNGLFRAMEBUFFERTEXTURE1DPROC __glewFramebufferTexture1D = NULL;
PFNGLFRAMEBUFFERTEXTURE2DPROC __glewFramebufferTexture2D = NULL;
PFNGLFRAMEBUFFERTEXTURE3DPROC __glewFramebufferTexture3D = NULL;
PFNGLFRAMEBUFFERTEXTURELAYERPROC __glewFramebufferTextureLayer = NULL;
PFNGLGENFRAMEBUFFERSPROC __glewGenFramebuffers = NULL;
PFNGLGENRENDERBUFFERSPROC __glewGenRenderbuffers = NULL;
PFNGLGENERATEMIPMAPPROC __glewGenerateMipmap = NULL;
PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC __glewGetFramebufferAttachmentParameteriv = NULL;
PFNGLGETRENDERBUFFERPARAMETERIVPROC __glewGetRenderbufferParameteriv = NULL;
PFNGLISFRAMEBUFFERPROC __glewIsFramebuffer = NULL;
PFNGLISRENDERBUFFERPROC __glewIsRenderbuffer = NULL;
PFNGLRENDERBUFFERSTORAGEPROC __glewRenderbufferStorage = NULL;
PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC __glewRenderbufferStorageMultisample = NULL;

PFNGLFRAMEBUFFERTEXTUREARBPROC __glewFramebufferTextureARB = NULL;
PFNGLFRAMEBUFFERTEXTUREFACEARBPROC __glewFramebufferTextureFaceARB = NULL;
PFNGLFRAMEBUFFERTEXTURELAYERARBPROC __glewFramebufferTextureLayerARB = NULL;
PFNGLPROGRAMPARAMETERIARBPROC __glewProgramParameteriARB = NULL;

PFNGLGETPROGRAMBINARYPROC __glewGetProgramBinary = NULL;
PFNGLPROGRAMBINARYPROC __glewProgramBinary = NULL;
PFNGLPROGRAMPARAMETERIPROC __glewProgramParameteri = NULL;

PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC __glewGetCompressedTextureSubImage = NULL;
PFNGLGETTEXTURESUBIMAGEPROC __glewGetTextureSubImage = NULL;

PFNGLGETUNIFORMDVPROC __glewGetUniformdv = NULL;
PFNGLUNIFORM1DPROC __glewUniform1d = NULL;
PFNGLUNIFORM1DVPROC __glewUniform1dv = NULL;
PFNGLUNIFORM2DPROC __glewUniform2d = NULL;
PFNGLUNIFORM2DVPROC __glewUniform2dv = NULL;
PFNGLUNIFORM3DPROC __glewUniform3d = NULL;
PFNGLUNIFORM3DVPROC __glewUniform3dv = NULL;
PFNGLUNIFORM4DPROC __glewUniform4d = NULL;
PFNGLUNIFORM4DVPROC __glewUniform4dv = NULL;
PFNGLUNIFORMMATRIX2DVPROC __glewUniformMatrix2dv = NULL;
PFNGLUNIFORMMATRIX2X3DVPROC __glewUniformMatrix2x3dv = NULL;
PFNGLUNIFORMMATRIX2X4DVPROC __glewUniformMatrix2x4dv = NULL;
PFNGLUNIFORMMATRIX3DVPROC __glewUniformMatrix3dv = NULL;
PFNGLUNIFORMMATRIX3X2DVPROC __glewUniformMatrix3x2dv = NULL;
PFNGLUNIFORMMATRIX3X4DVPROC __glewUniformMatrix3x4dv = NULL;
PFNGLUNIFORMMATRIX4DVPROC __glewUniformMatrix4dv = NULL;
PFNGLUNIFORMMATRIX4X2DVPROC __glewUniformMatrix4x2dv = NULL;
PFNGLUNIFORMMATRIX4X3DVPROC __glewUniformMatrix4x3dv = NULL;

PFNGLGETUNIFORMI64VARBPROC __glewGetUniformi64vARB = NULL;
PFNGLGETUNIFORMUI64VARBPROC __glewGetUniformui64vARB = NULL;
PFNGLGETNUNIFORMI64VARBPROC __glewGetnUniformi64vARB = NULL;
PFNGLGETNUNIFORMUI64VARBPROC __glewGetnUniformui64vARB = NULL;
PFNGLPROGRAMUNIFORM1I64ARBPROC __glewProgramUniform1i64ARB = NULL;
PFNGLPROGRAMUNIFORM1I64VARBPROC __glewProgramUniform1i64vARB = NULL;
PFNGLPROGRAMUNIFORM1UI64ARBPROC __glewProgramUniform1ui64ARB = NULL;
PFNGLPROGRAMUNIFORM1UI64VARBPROC __glewProgramUniform1ui64vARB = NULL;
PFNGLPROGRAMUNIFORM2I64ARBPROC __glewProgramUniform2i64ARB = NULL;
PFNGLPROGRAMUNIFORM2I64VARBPROC __glewProgramUniform2i64vARB = NULL;
PFNGLPROGRAMUNIFORM2UI64ARBPROC __glewProgramUniform2ui64ARB = NULL;
PFNGLPROGRAMUNIFORM2UI64VARBPROC __glewProgramUniform2ui64vARB = NULL;
PFNGLPROGRAMUNIFORM3I64ARBPROC __glewProgramUniform3i64ARB = NULL;
PFNGLPROGRAMUNIFORM3I64VARBPROC __glewProgramUniform3i64vARB = NULL;
PFNGLPROGRAMUNIFORM3UI64ARBPROC __glewProgramUniform3ui64ARB = NULL;
PFNGLPROGRAMUNIFORM3UI64VARBPROC __glewProgramUniform3ui64vARB = NULL;
PFNGLPROGRAMUNIFORM4I64ARBPROC __glewProgramUniform4i64ARB = NULL;
PFNGLPROGRAMUNIFORM4I64VARBPROC __glewProgramUniform4i64vARB = NULL;
PFNGLPROGRAMUNIFORM4UI64ARBPROC __glewProgramUniform4ui64ARB = NULL;
PFNGLPROGRAMUNIFORM4UI64VARBPROC __glewProgramUniform4ui64vARB = NULL;
PFNGLUNIFORM1I64ARBPROC __glewUniform1i64ARB = NULL;
PFNGLUNIFORM1I64VARBPROC __glewUniform1i64vARB = NULL;
PFNGLUNIFORM1UI64ARBPROC __glewUniform1ui64ARB = NULL;
PFNGLUNIFORM1UI64VARBPROC __glewUniform1ui64vARB = NULL;
PFNGLUNIFORM2I64ARBPROC __glewUniform2i64ARB = NULL;
PFNGLUNIFORM2I64VARBPROC __glewUniform2i64vARB = NULL;
PFNGLUNIFORM2UI64ARBPROC __glewUniform2ui64ARB = NULL;
PFNGLUNIFORM2UI64VARBPROC __glewUniform2ui64vARB = NULL;
PFNGLUNIFORM3I64ARBPROC __glewUniform3i64ARB = NULL;
PFNGLUNIFORM3I64VARBPROC __glewUniform3i64vARB = NULL;
PFNGLUNIFORM3UI64ARBPROC __glewUniform3ui64ARB = NULL;
PFNGLUNIFORM3UI64VARBPROC __glewUniform3ui64vARB = NULL;
PFNGLUNIFORM4I64ARBPROC __glewUniform4i64ARB = NULL;
PFNGLUNIFORM4I64VARBPROC __glewUniform4i64vARB = NULL;
PFNGLUNIFORM4UI64ARBPROC __glewUniform4ui64ARB = NULL;
PFNGLUNIFORM4UI64VARBPROC __glewUniform4ui64vARB = NULL;

PFNGLCOLORSUBTABLEPROC __glewColorSubTable = NULL;
PFNGLCOLORTABLEPROC __glewColorTable = NULL;
PFNGLCOLORTABLEPARAMETERFVPROC __glewColorTableParameterfv = NULL;
PFNGLCOLORTABLEPARAMETERIVPROC __glewColorTableParameteriv = NULL;
PFNGLCONVOLUTIONFILTER1DPROC __glewConvolutionFilter1D = NULL;
PFNGLCONVOLUTIONFILTER2DPROC __glewConvolutionFilter2D = NULL;
PFNGLCONVOLUTIONPARAMETERFPROC __glewConvolutionParameterf = NULL;
PFNGLCONVOLUTIONPARAMETERFVPROC __glewConvolutionParameterfv = NULL;
PFNGLCONVOLUTIONPARAMETERIPROC __glewConvolutionParameteri = NULL;
PFNGLCONVOLUTIONPARAMETERIVPROC __glewConvolutionParameteriv = NULL;
PFNGLCOPYCOLORSUBTABLEPROC __glewCopyColorSubTable = NULL;
PFNGLCOPYCOLORTABLEPROC __glewCopyColorTable = NULL;
PFNGLCOPYCONVOLUTIONFILTER1DPROC __glewCopyConvolutionFilter1D = NULL;
PFNGLCOPYCONVOLUTIONFILTER2DPROC __glewCopyConvolutionFilter2D = NULL;
PFNGLGETCOLORTABLEPROC __glewGetColorTable = NULL;
PFNGLGETCOLORTABLEPARAMETERFVPROC __glewGetColorTableParameterfv = NULL;
PFNGLGETCOLORTABLEPARAMETERIVPROC __glewGetColorTableParameteriv = NULL;
PFNGLGETCONVOLUTIONFILTERPROC __glewGetConvolutionFilter = NULL;
PFNGLGETCONVOLUTIONPARAMETERFVPROC __glewGetConvolutionParameterfv = NULL;
PFNGLGETCONVOLUTIONPARAMETERIVPROC __glewGetConvolutionParameteriv = NULL;
PFNGLGETHISTOGRAMPROC __glewGetHistogram = NULL;
PFNGLGETHISTOGRAMPARAMETERFVPROC __glewGetHistogramParameterfv = NULL;
PFNGLGETHISTOGRAMPARAMETERIVPROC __glewGetHistogramParameteriv = NULL;
PFNGLGETMINMAXPROC __glewGetMinmax = NULL;
PFNGLGETMINMAXPARAMETERFVPROC __glewGetMinmaxParameterfv = NULL;
PFNGLGETMINMAXPARAMETERIVPROC __glewGetMinmaxParameteriv = NULL;
PFNGLGETSEPARABLEFILTERPROC __glewGetSeparableFilter = NULL;
PFNGLHISTOGRAMPROC __glewHistogram = NULL;
PFNGLMINMAXPROC __glewMinmax = NULL;
PFNGLRESETHISTOGRAMPROC __glewResetHistogram = NULL;
PFNGLRESETMINMAXPROC __glewResetMinmax = NULL;
PFNGLSEPARABLEFILTER2DPROC __glewSeparableFilter2D = NULL;

PFNGLMULTIDRAWARRAYSINDIRECTCOUNTARBPROC __glewMultiDrawArraysIndirectCountARB = NULL;
PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTARBPROC __glewMultiDrawElementsIndirectCountARB = NULL;

PFNGLDRAWARRAYSINSTANCEDARBPROC __glewDrawArraysInstancedARB = NULL;
PFNGLDRAWELEMENTSINSTANCEDARBPROC __glewDrawElementsInstancedARB = NULL;
PFNGLVERTEXATTRIBDIVISORARBPROC __glewVertexAttribDivisorARB = NULL;

PFNGLGETINTERNALFORMATIVPROC __glewGetInternalformativ = NULL;

PFNGLGETINTERNALFORMATI64VPROC __glewGetInternalformati64v = NULL;

PFNGLINVALIDATEBUFFERDATAPROC __glewInvalidateBufferData = NULL;
PFNGLINVALIDATEBUFFERSUBDATAPROC __glewInvalidateBufferSubData = NULL;
PFNGLINVALIDATEFRAMEBUFFERPROC __glewInvalidateFramebuffer = NULL;
PFNGLINVALIDATESUBFRAMEBUFFERPROC __glewInvalidateSubFramebuffer = NULL;
PFNGLINVALIDATETEXIMAGEPROC __glewInvalidateTexImage = NULL;
PFNGLINVALIDATETEXSUBIMAGEPROC __glewInvalidateTexSubImage = NULL;

PFNGLFLUSHMAPPEDBUFFERRANGEPROC __glewFlushMappedBufferRange = NULL;
PFNGLMAPBUFFERRANGEPROC __glewMapBufferRange = NULL;

PFNGLCURRENTPALETTEMATRIXARBPROC __glewCurrentPaletteMatrixARB = NULL;
PFNGLMATRIXINDEXPOINTERARBPROC __glewMatrixIndexPointerARB = NULL;
PFNGLMATRIXINDEXUBVARBPROC __glewMatrixIndexubvARB = NULL;
PFNGLMATRIXINDEXUIVARBPROC __glewMatrixIndexuivARB = NULL;
PFNGLMATRIXINDEXUSVARBPROC __glewMatrixIndexusvARB = NULL;

PFNGLBINDBUFFERSBASEPROC __glewBindBuffersBase = NULL;
PFNGLBINDBUFFERSRANGEPROC __glewBindBuffersRange = NULL;
PFNGLBINDIMAGETEXTURESPROC __glewBindImageTextures = NULL;
PFNGLBINDSAMPLERSPROC __glewBindSamplers = NULL;
PFNGLBINDTEXTURESPROC __glewBindTextures = NULL;
PFNGLBINDVERTEXBUFFERSPROC __glewBindVertexBuffers = NULL;

PFNGLMULTIDRAWARRAYSINDIRECTPROC __glewMultiDrawArraysIndirect = NULL;
PFNGLMULTIDRAWELEMENTSINDIRECTPROC __glewMultiDrawElementsIndirect = NULL;

PFNGLSAMPLECOVERAGEARBPROC __glewSampleCoverageARB = NULL;

PFNGLACTIVETEXTUREARBPROC __glewActiveTextureARB = NULL;
PFNGLCLIENTACTIVETEXTUREARBPROC __glewClientActiveTextureARB = NULL;
PFNGLMULTITEXCOORD1DARBPROC __glewMultiTexCoord1dARB = NULL;
PFNGLMULTITEXCOORD1DVARBPROC __glewMultiTexCoord1dvARB = NULL;
PFNGLMULTITEXCOORD1FARBPROC __glewMultiTexCoord1fARB = NULL;
PFNGLMULTITEXCOORD1FVARBPROC __glewMultiTexCoord1fvARB = NULL;
PFNGLMULTITEXCOORD1IARBPROC __glewMultiTexCoord1iARB = NULL;
PFNGLMULTITEXCOORD1IVARBPROC __glewMultiTexCoord1ivARB = NULL;
PFNGLMULTITEXCOORD1SARBPROC __glewMultiTexCoord1sARB = NULL;
PFNGLMULTITEXCOORD1SVARBPROC __glewMultiTexCoord1svARB = NULL;
PFNGLMULTITEXCOORD2DARBPROC __glewMultiTexCoord2dARB = NULL;
PFNGLMULTITEXCOORD2DVARBPROC __glewMultiTexCoord2dvARB = NULL;
PFNGLMULTITEXCOORD2FARBPROC __glewMultiTexCoord2fARB = NULL;
PFNGLMULTITEXCOORD2FVARBPROC __glewMultiTexCoord2fvARB = NULL;
PFNGLMULTITEXCOORD2IARBPROC __glewMultiTexCoord2iARB = NULL;
PFNGLMULTITEXCOORD2IVARBPROC __glewMultiTexCoord2ivARB = NULL;
PFNGLMULTITEXCOORD2SARBPROC __glewMultiTexCoord2sARB = NULL;
PFNGLMULTITEXCOORD2SVARBPROC __glewMultiTexCoord2svARB = NULL;
PFNGLMULTITEXCOORD3DARBPROC __glewMultiTexCoord3dARB = NULL;
PFNGLMULTITEXCOORD3DVARBPROC __glewMultiTexCoord3dvARB = NULL;
PFNGLMULTITEXCOORD3FARBPROC __glewMultiTexCoord3fARB = NULL;
PFNGLMULTITEXCOORD3FVARBPROC __glewMultiTexCoord3fvARB = NULL;
PFNGLMULTITEXCOORD3IARBPROC __glewMultiTexCoord3iARB = NULL;
PFNGLMULTITEXCOORD3IVARBPROC __glewMultiTexCoord3ivARB = NULL;
PFNGLMULTITEXCOORD3SARBPROC __glewMultiTexCoord3sARB = NULL;
PFNGLMULTITEXCOORD3SVARBPROC __glewMultiTexCoord3svARB = NULL;
PFNGLMULTITEXCOORD4DARBPROC __glewMultiTexCoord4dARB = NULL;
PFNGLMULTITEXCOORD4DVARBPROC __glewMultiTexCoord4dvARB = NULL;
PFNGLMULTITEXCOORD4FARBPROC __glewMultiTexCoord4fARB = NULL;
PFNGLMULTITEXCOORD4FVARBPROC __glewMultiTexCoord4fvARB = NULL;
PFNGLMULTITEXCOORD4IARBPROC __glewMultiTexCoord4iARB = NULL;
PFNGLMULTITEXCOORD4IVARBPROC __glewMultiTexCoord4ivARB = NULL;
PFNGLMULTITEXCOORD4SARBPROC __glewMultiTexCoord4sARB = NULL;
PFNGLMULTITEXCOORD4SVARBPROC __glewMultiTexCoord4svARB = NULL;

PFNGLBEGINQUERYARBPROC __glewBeginQueryARB = NULL;
PFNGLDELETEQUERIESARBPROC __glewDeleteQueriesARB = NULL;
PFNGLENDQUERYARBPROC __glewEndQueryARB = NULL;
PFNGLGENQUERIESARBPROC __glewGenQueriesARB = NULL;
PFNGLGETQUERYOBJECTIVARBPROC __glewGetQueryObjectivARB = NULL;
PFNGLGETQUERYOBJECTUIVARBPROC __glewGetQueryObjectuivARB = NULL;
PFNGLGETQUERYIVARBPROC __glewGetQueryivARB = NULL;
PFNGLISQUERYARBPROC __glewIsQueryARB = NULL;

PFNGLMAXSHADERCOMPILERTHREADSARBPROC __glewMaxShaderCompilerThreadsARB = NULL;

PFNGLPOINTPARAMETERFARBPROC __glewPointParameterfARB = NULL;
PFNGLPOINTPARAMETERFVARBPROC __glewPointParameterfvARB = NULL;

PFNGLGETPROGRAMINTERFACEIVPROC __glewGetProgramInterfaceiv = NULL;
PFNGLGETPROGRAMRESOURCEINDEXPROC __glewGetProgramResourceIndex = NULL;
PFNGLGETPROGRAMRESOURCELOCATIONPROC __glewGetProgramResourceLocation = NULL;
PFNGLGETPROGRAMRESOURCELOCATIONINDEXPROC __glewGetProgramResourceLocationIndex = NULL;
PFNGLGETPROGRAMRESOURCENAMEPROC __glewGetProgramResourceName = NULL;
PFNGLGETPROGRAMRESOURCEIVPROC __glewGetProgramResourceiv = NULL;

PFNGLPROVOKINGVERTEXPROC __glewProvokingVertex = NULL;

PFNGLGETGRAPHICSRESETSTATUSARBPROC __glewGetGraphicsResetStatusARB = NULL;
PFNGLGETNCOLORTABLEARBPROC __glewGetnColorTableARB = NULL;
PFNGLGETNCOMPRESSEDTEXIMAGEARBPROC __glewGetnCompressedTexImageARB = NULL;
PFNGLGETNCONVOLUTIONFILTERARBPROC __glewGetnConvolutionFilterARB = NULL;
PFNGLGETNHISTOGRAMARBPROC __glewGetnHistogramARB = NULL;
PFNGLGETNMAPDVARBPROC __glewGetnMapdvARB = NULL;
PFNGLGETNMAPFVARBPROC __glewGetnMapfvARB = NULL;
PFNGLGETNMAPIVARBPROC __glewGetnMapivARB = NULL;
PFNGLGETNMINMAXARBPROC __glewGetnMinmaxARB = NULL;
PFNGLGETNPIXELMAPFVARBPROC __glewGetnPixelMapfvARB = NULL;
PFNGLGETNPIXELMAPUIVARBPROC __glewGetnPixelMapuivARB = NULL;
PFNGLGETNPIXELMAPUSVARBPROC __glewGetnPixelMapusvARB = NULL;
PFNGLGETNPOLYGONSTIPPLEARBPROC __glewGetnPolygonStippleARB = NULL;
PFNGLGETNSEPARABLEFILTERARBPROC __glewGetnSeparableFilterARB = NULL;
PFNGLGETNTEXIMAGEARBPROC __glewGetnTexImageARB = NULL;
PFNGLGETNUNIFORMDVARBPROC __glewGetnUniformdvARB = NULL;
PFNGLGETNUNIFORMFVARBPROC __glewGetnUniformfvARB = NULL;
PFNGLGETNUNIFORMIVARBPROC __glewGetnUniformivARB = NULL;
PFNGLGETNUNIFORMUIVARBPROC __glewGetnUniformuivARB = NULL;
PFNGLREADNPIXELSARBPROC __glewReadnPixelsARB = NULL;

PFNGLFRAMEBUFFERSAMPLELOCATIONSFVARBPROC __glewFramebufferSampleLocationsfvARB = NULL;
PFNGLNAMEDFRAMEBUFFERSAMPLELOCATIONSFVARBPROC __glewNamedFramebufferSampleLocationsfvARB = NULL;

PFNGLMINSAMPLESHADINGARBPROC __glewMinSampleShadingARB = NULL;

PFNGLBINDSAMPLERPROC __glewBindSampler = NULL;
PFNGLDELETESAMPLERSPROC __glewDeleteSamplers = NULL;
PFNGLGENSAMPLERSPROC __glewGenSamplers = NULL;
PFNGLGETSAMPLERPARAMETERIIVPROC __glewGetSamplerParameterIiv = NULL;
PFNGLGETSAMPLERPARAMETERIUIVPROC __glewGetSamplerParameterIuiv = NULL;
PFNGLGETSAMPLERPARAMETERFVPROC __glewGetSamplerParameterfv = NULL;
PFNGLGETSAMPLERPARAMETERIVPROC __glewGetSamplerParameteriv = NULL;
PFNGLISSAMPLERPROC __glewIsSampler = NULL;
PFNGLSAMPLERPARAMETERIIVPROC __glewSamplerParameterIiv = NULL;
PFNGLSAMPLERPARAMETERIUIVPROC __glewSamplerParameterIuiv = NULL;
PFNGLSAMPLERPARAMETERFPROC __glewSamplerParameterf = NULL;
PFNGLSAMPLERPARAMETERFVPROC __glewSamplerParameterfv = NULL;
PFNGLSAMPLERPARAMETERIPROC __glewSamplerParameteri = NULL;
PFNGLSAMPLERPARAMETERIVPROC __glewSamplerParameteriv = NULL;

PFNGLACTIVESHADERPROGRAMPROC __glewActiveShaderProgram = NULL;
PFNGLBINDPROGRAMPIPELINEPROC __glewBindProgramPipeline = NULL;
PFNGLCREATESHADERPROGRAMVPROC __glewCreateShaderProgramv = NULL;
PFNGLDELETEPROGRAMPIPELINESPROC __glewDeleteProgramPipelines = NULL;
PFNGLGENPROGRAMPIPELINESPROC __glewGenProgramPipelines = NULL;
PFNGLGETPROGRAMPIPELINEINFOLOGPROC __glewGetProgramPipelineInfoLog = NULL;
PFNGLGETPROGRAMPIPELINEIVPROC __glewGetProgramPipelineiv = NULL;
PFNGLISPROGRAMPIPELINEPROC __glewIsProgramPipeline = NULL;
PFNGLPROGRAMUNIFORM1DPROC __glewProgramUniform1d = NULL;
PFNGLPROGRAMUNIFORM1DVPROC __glewProgramUniform1dv = NULL;
PFNGLPROGRAMUNIFORM1FPROC __glewProgramUniform1f = NULL;
PFNGLPROGRAMUNIFORM1FVPROC __glewProgramUniform1fv = NULL;
PFNGLPROGRAMUNIFORM1IPROC __glewProgramUniform1i = NULL;
PFNGLPROGRAMUNIFORM1IVPROC __glewProgramUniform1iv = NULL;
PFNGLPROGRAMUNIFORM1UIPROC __glewProgramUniform1ui = NULL;
PFNGLPROGRAMUNIFORM1UIVPROC __glewProgramUniform1uiv = NULL;
PFNGLPROGRAMUNIFORM2DPROC __glewProgramUniform2d = NULL;
PFNGLPROGRAMUNIFORM2DVPROC __glewProgramUniform2dv = NULL;
PFNGLPROGRAMUNIFORM2FPROC __glewProgramUniform2f = NULL;
PFNGLPROGRAMUNIFORM2FVPROC __glewProgramUniform2fv = NULL;
PFNGLPROGRAMUNIFORM2IPROC __glewProgramUniform2i = NULL;
PFNGLPROGRAMUNIFORM2IVPROC __glewProgramUniform2iv = NULL;
PFNGLPROGRAMUNIFORM2UIPROC __glewProgramUniform2ui = NULL;
PFNGLPROGRAMUNIFORM2UIVPROC __glewProgramUniform2uiv = NULL;
PFNGLPROGRAMUNIFORM3DPROC __glewProgramUniform3d = NULL;
PFNGLPROGRAMUNIFORM3DVPROC __glewProgramUniform3dv = NULL;
PFNGLPROGRAMUNIFORM3FPROC __glewProgramUniform3f = NULL;
PFNGLPROGRAMUNIFORM3FVPROC __glewProgramUniform3fv = NULL;
PFNGLPROGRAMUNIFORM3IPROC __glewProgramUniform3i = NULL;
PFNGLPROGRAMUNIFORM3IVPROC __glewProgramUniform3iv = NULL;
PFNGLPROGRAMUNIFORM3UIPROC __glewProgramUniform3ui = NULL;
PFNGLPROGRAMUNIFORM3UIVPROC __glewProgramUniform3uiv = NULL;
PFNGLPROGRAMUNIFORM4DPROC __glewProgramUniform4d = NULL;
PFNGLPROGRAMUNIFORM4DVPROC __glewProgramUniform4dv = NULL;
PFNGLPROGRAMUNIFORM4FPROC __glewProgramUniform4f = NULL;
PFNGLPROGRAMUNIFORM4FVPROC __glewProgramUniform4fv = NULL;
PFNGLPROGRAMUNIFORM4IPROC __glewProgramUniform4i = NULL;
PFNGLPROGRAMUNIFORM4IVPROC __glewProgramUniform4iv = NULL;
PFNGLPROGRAMUNIFORM4UIPROC __glewProgramUniform4ui = NULL;
PFNGLPROGRAMUNIFORM4UIVPROC __glewProgramUniform4uiv = NULL;
PFNGLPROGRAMUNIFORMMATRIX2DVPROC __glewProgramUniformMatrix2dv = NULL;
PFNGLPROGRAMUNIFORMMATRIX2FVPROC __glewProgramUniformMatrix2fv = NULL;
PFNGLPROGRAMUNIFORMMATRIX2X3DVPROC __glewProgramUniformMatrix2x3dv = NULL;
PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC __glewProgramUniformMatrix2x3fv = NULL;
PFNGLPROGRAMUNIFORMMATRIX2X4DVPROC __glewProgramUniformMatrix2x4dv = NULL;
PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC __glewProgramUniformMatrix2x4fv = NULL;
PFNGLPROGRAMUNIFORMMATRIX3DVPROC __glewProgramUniformMatrix3dv = NULL;
PFNGLPROGRAMUNIFORMMATRIX3FVPROC __glewProgramUniformMatrix3fv = NULL;
PFNGLPROGRAMUNIFORMMATRIX3X2DVPROC __glewProgramUniformMatrix3x2dv = NULL;
PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC __glewProgramUniformMatrix3x2fv = NULL;
PFNGLPROGRAMUNIFORMMATRIX3X4DVPROC __glewProgramUniformMatrix3x4dv = NULL;
PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC __glewProgramUniformMatrix3x4fv = NULL;
PFNGLPROGRAMUNIFORMMATRIX4DVPROC __glewProgramUniformMatrix4dv = NULL;
PFNGLPROGRAMUNIFORMMATRIX4FVPROC __glewProgramUniformMatrix4fv = NULL;
PFNGLPROGRAMUNIFORMMATRIX4X2DVPROC __glewProgramUniformMatrix4x2dv = NULL;
PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC __glewProgramUniformMatrix4x2fv = NULL;
PFNGLPROGRAMUNIFORMMATRIX4X3DVPROC __glewProgramUniformMatrix4x3dv = NULL;
PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC __glewProgramUniformMatrix4x3fv = NULL;
PFNGLUSEPROGRAMSTAGESPROC __glewUseProgramStages = NULL;
PFNGLVALIDATEPROGRAMPIPELINEPROC __glewValidateProgramPipeline = NULL;

PFNGLGETACTIVEATOMICCOUNTERBUFFERIVPROC __glewGetActiveAtomicCounterBufferiv = NULL;

PFNGLBINDIMAGETEXTUREPROC __glewBindImageTexture = NULL;
PFNGLMEMORYBARRIERPROC __glewMemoryBarrier = NULL;

PFNGLATTACHOBJECTARBPROC __glewAttachObjectARB = NULL;
PFNGLCOMPILESHADERARBPROC __glewCompileShaderARB = NULL;
PFNGLCREATEPROGRAMOBJECTARBPROC __glewCreateProgramObjectARB = NULL;
PFNGLCREATESHADEROBJECTARBPROC __glewCreateShaderObjectARB = NULL;
PFNGLDELETEOBJECTARBPROC __glewDeleteObjectARB = NULL;
PFNGLDETACHOBJECTARBPROC __glewDetachObjectARB = NULL;
PFNGLGETACTIVEUNIFORMARBPROC __glewGetActiveUniformARB = NULL;
PFNGLGETATTACHEDOBJECTSARBPROC __glewGetAttachedObjectsARB = NULL;
PFNGLGETHANDLEARBPROC __glewGetHandleARB = NULL;
PFNGLGETINFOLOGARBPROC __glewGetInfoLogARB = NULL;
PFNGLGETOBJECTPARAMETERFVARBPROC __glewGetObjectParameterfvARB = NULL;
PFNGLGETOBJECTPARAMETERIVARBPROC __glewGetObjectParameterivARB = NULL;
PFNGLGETSHADERSOURCEARBPROC __glewGetShaderSourceARB = NULL;
PFNGLGETUNIFORMLOCATIONARBPROC __glewGetUniformLocationARB = NULL;
PFNGLGETUNIFORMFVARBPROC __glewGetUniformfvARB = NULL;
PFNGLGETUNIFORMIVARBPROC __glewGetUniformivARB = NULL;
PFNGLLINKPROGRAMARBPROC __glewLinkProgramARB = NULL;
PFNGLSHADERSOURCEARBPROC __glewShaderSourceARB = NULL;
PFNGLUNIFORM1FARBPROC __glewUniform1fARB = NULL;
PFNGLUNIFORM1FVARBPROC __glewUniform1fvARB = NULL;
PFNGLUNIFORM1IARBPROC __glewUniform1iARB = NULL;
PFNGLUNIFORM1IVARBPROC __glewUniform1ivARB = NULL;
PFNGLUNIFORM2FARBPROC __glewUniform2fARB = NULL;
PFNGLUNIFORM2FVARBPROC __glewUniform2fvARB = NULL;
PFNGLUNIFORM2IARBPROC __glewUniform2iARB = NULL;
PFNGLUNIFORM2IVARBPROC __glewUniform2ivARB = NULL;
PFNGLUNIFORM3FARBPROC __glewUniform3fARB = NULL;
PFNGLUNIFORM3FVARBPROC __glewUniform3fvARB = NULL;
PFNGLUNIFORM3IARBPROC __glewUniform3iARB = NULL;
PFNGLUNIFORM3IVARBPROC __glewUniform3ivARB = NULL;
PFNGLUNIFORM4FARBPROC __glewUniform4fARB = NULL;
PFNGLUNIFORM4FVARBPROC __glewUniform4fvARB = NULL;
PFNGLUNIFORM4IARBPROC __glewUniform4iARB = NULL;
PFNGLUNIFORM4IVARBPROC __glewUniform4ivARB = NULL;
PFNGLUNIFORMMATRIX2FVARBPROC __glewUniformMatrix2fvARB = NULL;
PFNGLUNIFORMMATRIX3FVARBPROC __glewUniformMatrix3fvARB = NULL;
PFNGLUNIFORMMATRIX4FVARBPROC __glewUniformMatrix4fvARB = NULL;
PFNGLUSEPROGRAMOBJECTARBPROC __glewUseProgramObjectARB = NULL;
PFNGLVALIDATEPROGRAMARBPROC __glewValidateProgramARB = NULL;

PFNGLSHADERSTORAGEBLOCKBINDINGPROC __glewShaderStorageBlockBinding = NULL;

PFNGLGETACTIVESUBROUTINENAMEPROC __glewGetActiveSubroutineName = NULL;
PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC __glewGetActiveSubroutineUniformName = NULL;
PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC __glewGetActiveSubroutineUniformiv = NULL;
PFNGLGETPROGRAMSTAGEIVPROC __glewGetProgramStageiv = NULL;
PFNGLGETSUBROUTINEINDEXPROC __glewGetSubroutineIndex = NULL;
PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC __glewGetSubroutineUniformLocation = NULL;
PFNGLGETUNIFORMSUBROUTINEUIVPROC __glewGetUniformSubroutineuiv = NULL;
PFNGLUNIFORMSUBROUTINESUIVPROC __glewUniformSubroutinesuiv = NULL;

PFNGLCOMPILESHADERINCLUDEARBPROC __glewCompileShaderIncludeARB = NULL;
PFNGLDELETENAMEDSTRINGARBPROC __glewDeleteNamedStringARB = NULL;
PFNGLGETNAMEDSTRINGARBPROC __glewGetNamedStringARB = NULL;
PFNGLGETNAMEDSTRINGIVARBPROC __glewGetNamedStringivARB = NULL;
PFNGLISNAMEDSTRINGARBPROC __glewIsNamedStringARB = NULL;
PFNGLNAMEDSTRINGARBPROC __glewNamedStringARB = NULL;

PFNGLBUFFERPAGECOMMITMENTARBPROC __glewBufferPageCommitmentARB = NULL;

PFNGLTEXPAGECOMMITMENTARBPROC __glewTexPageCommitmentARB = NULL;
PFNGLTEXTUREPAGECOMMITMENTEXTPROC __glewTexturePageCommitmentEXT = NULL;

PFNGLCLIENTWAITSYNCPROC __glewClientWaitSync = NULL;
PFNGLDELETESYNCPROC __glewDeleteSync = NULL;
PFNGLFENCESYNCPROC __glewFenceSync = NULL;
PFNGLGETINTEGER64VPROC __glewGetInteger64v = NULL;
PFNGLGETSYNCIVPROC __glewGetSynciv = NULL;
PFNGLISSYNCPROC __glewIsSync = NULL;
PFNGLWAITSYNCPROC __glewWaitSync = NULL;

PFNGLPATCHPARAMETERFVPROC __glewPatchParameterfv = NULL;
PFNGLPATCHPARAMETERIPROC __glewPatchParameteri = NULL;

PFNGLTEXTUREBARRIERPROC __glewTextureBarrier = NULL;

PFNGLTEXBUFFERARBPROC __glewTexBufferARB = NULL;

PFNGLTEXBUFFERRANGEPROC __glewTexBufferRange = NULL;
PFNGLTEXTUREBUFFERRANGEEXTPROC __glewTextureBufferRangeEXT = NULL;

PFNGLCOMPRESSEDTEXIMAGE1DARBPROC __glewCompressedTexImage1DARB = NULL;
PFNGLCOMPRESSEDTEXIMAGE2DARBPROC __glewCompressedTexImage2DARB = NULL;
PFNGLCOMPRESSEDTEXIMAGE3DARBPROC __glewCompressedTexImage3DARB = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC __glewCompressedTexSubImage1DARB = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC __glewCompressedTexSubImage2DARB = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC __glewCompressedTexSubImage3DARB = NULL;
PFNGLGETCOMPRESSEDTEXIMAGEARBPROC __glewGetCompressedTexImageARB = NULL;

PFNGLGETMULTISAMPLEFVPROC __glewGetMultisamplefv = NULL;
PFNGLSAMPLEMASKIPROC __glewSampleMaski = NULL;
PFNGLTEXIMAGE2DMULTISAMPLEPROC __glewTexImage2DMultisample = NULL;
PFNGLTEXIMAGE3DMULTISAMPLEPROC __glewTexImage3DMultisample = NULL;

PFNGLTEXSTORAGE1DPROC __glewTexStorage1D = NULL;
PFNGLTEXSTORAGE2DPROC __glewTexStorage2D = NULL;
PFNGLTEXSTORAGE3DPROC __glewTexStorage3D = NULL;
PFNGLTEXTURESTORAGE1DEXTPROC __glewTextureStorage1DEXT = NULL;
PFNGLTEXTURESTORAGE2DEXTPROC __glewTextureStorage2DEXT = NULL;
PFNGLTEXTURESTORAGE3DEXTPROC __glewTextureStorage3DEXT = NULL;

PFNGLTEXSTORAGE2DMULTISAMPLEPROC __glewTexStorage2DMultisample = NULL;
PFNGLTEXSTORAGE3DMULTISAMPLEPROC __glewTexStorage3DMultisample = NULL;
PFNGLTEXTURESTORAGE2DMULTISAMPLEEXTPROC __glewTextureStorage2DMultisampleEXT = NULL;
PFNGLTEXTURESTORAGE3DMULTISAMPLEEXTPROC __glewTextureStorage3DMultisampleEXT = NULL;

PFNGLTEXTUREVIEWPROC __glewTextureView = NULL;

PFNGLGETQUERYOBJECTI64VPROC __glewGetQueryObjecti64v = NULL;
PFNGLGETQUERYOBJECTUI64VPROC __glewGetQueryObjectui64v = NULL;
PFNGLQUERYCOUNTERPROC __glewQueryCounter = NULL;

PFNGLBINDTRANSFORMFEEDBACKPROC __glewBindTransformFeedback = NULL;
PFNGLDELETETRANSFORMFEEDBACKSPROC __glewDeleteTransformFeedbacks = NULL;
PFNGLDRAWTRANSFORMFEEDBACKPROC __glewDrawTransformFeedback = NULL;
PFNGLGENTRANSFORMFEEDBACKSPROC __glewGenTransformFeedbacks = NULL;
PFNGLISTRANSFORMFEEDBACKPROC __glewIsTransformFeedback = NULL;
PFNGLPAUSETRANSFORMFEEDBACKPROC __glewPauseTransformFeedback = NULL;
PFNGLRESUMETRANSFORMFEEDBACKPROC __glewResumeTransformFeedback = NULL;

PFNGLBEGINQUERYINDEXEDPROC __glewBeginQueryIndexed = NULL;
PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC __glewDrawTransformFeedbackStream = NULL;
PFNGLENDQUERYINDEXEDPROC __glewEndQueryIndexed = NULL;
PFNGLGETQUERYINDEXEDIVPROC __glewGetQueryIndexediv = NULL;

PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC __glewDrawTransformFeedbackInstanced = NULL;
PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC __glewDrawTransformFeedbackStreamInstanced = NULL;

PFNGLLOADTRANSPOSEMATRIXDARBPROC __glewLoadTransposeMatrixdARB = NULL;
PFNGLLOADTRANSPOSEMATRIXFARBPROC __glewLoadTransposeMatrixfARB = NULL;
PFNGLMULTTRANSPOSEMATRIXDARBPROC __glewMultTransposeMatrixdARB = NULL;
PFNGLMULTTRANSPOSEMATRIXFARBPROC __glewMultTransposeMatrixfARB = NULL;

PFNGLBINDBUFFERBASEPROC __glewBindBufferBase = NULL;
PFNGLBINDBUFFERRANGEPROC __glewBindBufferRange = NULL;
PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC __glewGetActiveUniformBlockName = NULL;
PFNGLGETACTIVEUNIFORMBLOCKIVPROC __glewGetActiveUniformBlockiv = NULL;
PFNGLGETACTIVEUNIFORMNAMEPROC __glewGetActiveUniformName = NULL;
PFNGLGETACTIVEUNIFORMSIVPROC __glewGetActiveUniformsiv = NULL;
PFNGLGETINTEGERI_VPROC __glewGetIntegeri_v = NULL;
PFNGLGETUNIFORMBLOCKINDEXPROC __glewGetUniformBlockIndex = NULL;
PFNGLGETUNIFORMINDICESPROC __glewGetUniformIndices = NULL;
PFNGLUNIFORMBLOCKBINDINGPROC __glewUniformBlockBinding = NULL;

PFNGLBINDVERTEXARRAYPROC __glewBindVertexArray = NULL;
PFNGLDELETEVERTEXARRAYSPROC __glewDeleteVertexArrays = NULL;
PFNGLGENVERTEXARRAYSPROC __glewGenVertexArrays = NULL;
PFNGLISVERTEXARRAYPROC __glewIsVertexArray = NULL;

PFNGLGETVERTEXATTRIBLDVPROC __glewGetVertexAttribLdv = NULL;
PFNGLVERTEXATTRIBL1DPROC __glewVertexAttribL1d = NULL;
PFNGLVERTEXATTRIBL1DVPROC __glewVertexAttribL1dv = NULL;
PFNGLVERTEXATTRIBL2DPROC __glewVertexAttribL2d = NULL;
PFNGLVERTEXATTRIBL2DVPROC __glewVertexAttribL2dv = NULL;
PFNGLVERTEXATTRIBL3DPROC __glewVertexAttribL3d = NULL;
PFNGLVERTEXATTRIBL3DVPROC __glewVertexAttribL3dv = NULL;
PFNGLVERTEXATTRIBL4DPROC __glewVertexAttribL4d = NULL;
PFNGLVERTEXATTRIBL4DVPROC __glewVertexAttribL4dv = NULL;
PFNGLVERTEXATTRIBLPOINTERPROC __glewVertexAttribLPointer = NULL;

PFNGLBINDVERTEXBUFFERPROC __glewBindVertexBuffer = NULL;
PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC __glewVertexArrayBindVertexBufferEXT = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC __glewVertexArrayVertexAttribBindingEXT = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC __glewVertexArrayVertexAttribFormatEXT = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC __glewVertexArrayVertexAttribIFormatEXT = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC __glewVertexArrayVertexAttribLFormatEXT = NULL;
PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC __glewVertexArrayVertexBindingDivisorEXT = NULL;
PFNGLVERTEXATTRIBBINDINGPROC __glewVertexAttribBinding = NULL;
PFNGLVERTEXATTRIBFORMATPROC __glewVertexAttribFormat = NULL;
PFNGLVERTEXATTRIBIFORMATPROC __glewVertexAttribIFormat = NULL;
PFNGLVERTEXATTRIBLFORMATPROC __glewVertexAttribLFormat = NULL;
PFNGLVERTEXBINDINGDIVISORPROC __glewVertexBindingDivisor = NULL;

PFNGLVERTEXBLENDARBPROC __glewVertexBlendARB = NULL;
PFNGLWEIGHTPOINTERARBPROC __glewWeightPointerARB = NULL;
PFNGLWEIGHTBVARBPROC __glewWeightbvARB = NULL;
PFNGLWEIGHTDVARBPROC __glewWeightdvARB = NULL;
PFNGLWEIGHTFVARBPROC __glewWeightfvARB = NULL;
PFNGLWEIGHTIVARBPROC __glewWeightivARB = NULL;
PFNGLWEIGHTSVARBPROC __glewWeightsvARB = NULL;
PFNGLWEIGHTUBVARBPROC __glewWeightubvARB = NULL;
PFNGLWEIGHTUIVARBPROC __glewWeightuivARB = NULL;
PFNGLWEIGHTUSVARBPROC __glewWeightusvARB = NULL;

PFNGLBINDBUFFERARBPROC __glewBindBufferARB = NULL;
PFNGLBUFFERDATAARBPROC __glewBufferDataARB = NULL;
PFNGLBUFFERSUBDATAARBPROC __glewBufferSubDataARB = NULL;
PFNGLDELETEBUFFERSARBPROC __glewDeleteBuffersARB = NULL;
PFNGLGENBUFFERSARBPROC __glewGenBuffersARB = NULL;
PFNGLGETBUFFERPARAMETERIVARBPROC __glewGetBufferParameterivARB = NULL;
PFNGLGETBUFFERPOINTERVARBPROC __glewGetBufferPointervARB = NULL;
PFNGLGETBUFFERSUBDATAARBPROC __glewGetBufferSubDataARB = NULL;
PFNGLISBUFFERARBPROC __glewIsBufferARB = NULL;
PFNGLMAPBUFFERARBPROC __glewMapBufferARB = NULL;
PFNGLUNMAPBUFFERARBPROC __glewUnmapBufferARB = NULL;

PFNGLBINDPROGRAMARBPROC __glewBindProgramARB = NULL;
PFNGLDELETEPROGRAMSARBPROC __glewDeleteProgramsARB = NULL;
PFNGLDISABLEVERTEXATTRIBARRAYARBPROC __glewDisableVertexAttribArrayARB = NULL;
PFNGLENABLEVERTEXATTRIBARRAYARBPROC __glewEnableVertexAttribArrayARB = NULL;
PFNGLGENPROGRAMSARBPROC __glewGenProgramsARB = NULL;
PFNGLGETPROGRAMENVPARAMETERDVARBPROC __glewGetProgramEnvParameterdvARB = NULL;
PFNGLGETPROGRAMENVPARAMETERFVARBPROC __glewGetProgramEnvParameterfvARB = NULL;
PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC __glewGetProgramLocalParameterdvARB = NULL;
PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC __glewGetProgramLocalParameterfvARB = NULL;
PFNGLGETPROGRAMSTRINGARBPROC __glewGetProgramStringARB = NULL;
PFNGLGETPROGRAMIVARBPROC __glewGetProgramivARB = NULL;
PFNGLGETVERTEXATTRIBPOINTERVARBPROC __glewGetVertexAttribPointervARB = NULL;
PFNGLGETVERTEXATTRIBDVARBPROC __glewGetVertexAttribdvARB = NULL;
PFNGLGETVERTEXATTRIBFVARBPROC __glewGetVertexAttribfvARB = NULL;
PFNGLGETVERTEXATTRIBIVARBPROC __glewGetVertexAttribivARB = NULL;
PFNGLISPROGRAMARBPROC __glewIsProgramARB = NULL;
PFNGLPROGRAMENVPARAMETER4DARBPROC __glewProgramEnvParameter4dARB = NULL;
PFNGLPROGRAMENVPARAMETER4DVARBPROC __glewProgramEnvParameter4dvARB = NULL;
PFNGLPROGRAMENVPARAMETER4FARBPROC __glewProgramEnvParameter4fARB = NULL;
PFNGLPROGRAMENVPARAMETER4FVARBPROC __glewProgramEnvParameter4fvARB = NULL;
PFNGLPROGRAMLOCALPARAMETER4DARBPROC __glewProgramLocalParameter4dARB = NULL;
PFNGLPROGRAMLOCALPARAMETER4DVARBPROC __glewProgramLocalParameter4dvARB = NULL;
PFNGLPROGRAMLOCALPARAMETER4FARBPROC __glewProgramLocalParameter4fARB = NULL;
PFNGLPROGRAMLOCALPARAMETER4FVARBPROC __glewProgramLocalParameter4fvARB = NULL;
PFNGLPROGRAMSTRINGARBPROC __glewProgramStringARB = NULL;
PFNGLVERTEXATTRIB1DARBPROC __glewVertexAttrib1dARB = NULL;
PFNGLVERTEXATTRIB1DVARBPROC __glewVertexAttrib1dvARB = NULL;
PFNGLVERTEXATTRIB1FARBPROC __glewVertexAttrib1fARB = NULL;
PFNGLVERTEXATTRIB1FVARBPROC __glewVertexAttrib1fvARB = NULL;
PFNGLVERTEXATTRIB1SARBPROC __glewVertexAttrib1sARB = NULL;
PFNGLVERTEXATTRIB1SVARBPROC __glewVertexAttrib1svARB = NULL;
PFNGLVERTEXATTRIB2DARBPROC __glewVertexAttrib2dARB = NULL;
PFNGLVERTEXATTRIB2DVARBPROC __glewVertexAttrib2dvARB = NULL;
PFNGLVERTEXATTRIB2FARBPROC __glewVertexAttrib2fARB = NULL;
PFNGLVERTEXATTRIB2FVARBPROC __glewVertexAttrib2fvARB = NULL;
PFNGLVERTEXATTRIB2SARBPROC __glewVertexAttrib2sARB = NULL;
PFNGLVERTEXATTRIB2SVARBPROC __glewVertexAttrib2svARB = NULL;
PFNGLVERTEXATTRIB3DARBPROC __glewVertexAttrib3dARB = NULL;
PFNGLVERTEXATTRIB3DVARBPROC __glewVertexAttrib3dvARB = NULL;
PFNGLVERTEXATTRIB3FARBPROC __glewVertexAttrib3fARB = NULL;
PFNGLVERTEXATTRIB3FVARBPROC __glewVertexAttrib3fvARB = NULL;
PFNGLVERTEXATTRIB3SARBPROC __glewVertexAttrib3sARB = NULL;
PFNGLVERTEXATTRIB3SVARBPROC __glewVertexAttrib3svARB = NULL;
PFNGLVERTEXATTRIB4NBVARBPROC __glewVertexAttrib4NbvARB = NULL;
PFNGLVERTEXATTRIB4NIVARBPROC __glewVertexAttrib4NivARB = NULL;
PFNGLVERTEXATTRIB4NSVARBPROC __glewVertexAttrib4NsvARB = NULL;
PFNGLVERTEXATTRIB4NUBARBPROC __glewVertexAttrib4NubARB = NULL;
PFNGLVERTEXATTRIB4NUBVARBPROC __glewVertexAttrib4NubvARB = NULL;
PFNGLVERTEXATTRIB4NUIVARBPROC __glewVertexAttrib4NuivARB = NULL;
PFNGLVERTEXATTRIB4NUSVARBPROC __glewVertexAttrib4NusvARB = NULL;
PFNGLVERTEXATTRIB4BVARBPROC __glewVertexAttrib4bvARB = NULL;
PFNGLVERTEXATTRIB4DARBPROC __glewVertexAttrib4dARB = NULL;
PFNGLVERTEXATTRIB4DVARBPROC __glewVertexAttrib4dvARB = NULL;
PFNGLVERTEXATTRIB4FARBPROC __glewVertexAttrib4fARB = NULL;
PFNGLVERTEXATTRIB4FVARBPROC __glewVertexAttrib4fvARB = NULL;
PFNGLVERTEXATTRIB4IVARBPROC __glewVertexAttrib4ivARB = NULL;
PFNGLVERTEXATTRIB4SARBPROC __glewVertexAttrib4sARB = NULL;
PFNGLVERTEXATTRIB4SVARBPROC __glewVertexAttrib4svARB = NULL;
PFNGLVERTEXATTRIB4UBVARBPROC __glewVertexAttrib4ubvARB = NULL;
PFNGLVERTEXATTRIB4UIVARBPROC __glewVertexAttrib4uivARB = NULL;
PFNGLVERTEXATTRIB4USVARBPROC __glewVertexAttrib4usvARB = NULL;
PFNGLVERTEXATTRIBPOINTERARBPROC __glewVertexAttribPointerARB = NULL;

PFNGLBINDATTRIBLOCATIONARBPROC __glewBindAttribLocationARB = NULL;
PFNGLGETACTIVEATTRIBARBPROC __glewGetActiveAttribARB = NULL;
PFNGLGETATTRIBLOCATIONARBPROC __glewGetAttribLocationARB = NULL;

PFNGLCOLORP3UIPROC __glewColorP3ui = NULL;
PFNGLCOLORP3UIVPROC __glewColorP3uiv = NULL;
PFNGLCOLORP4UIPROC __glewColorP4ui = NULL;
PFNGLCOLORP4UIVPROC __glewColorP4uiv = NULL;
PFNGLMULTITEXCOORDP1UIPROC __glewMultiTexCoordP1ui = NULL;
PFNGLMULTITEXCOORDP1UIVPROC __glewMultiTexCoordP1uiv = NULL;
PFNGLMULTITEXCOORDP2UIPROC __glewMultiTexCoordP2ui = NULL;
PFNGLMULTITEXCOORDP2UIVPROC __glewMultiTexCoordP2uiv = NULL;
PFNGLMULTITEXCOORDP3UIPROC __glewMultiTexCoordP3ui = NULL;
PFNGLMULTITEXCOORDP3UIVPROC __glewMultiTexCoordP3uiv = NULL;
PFNGLMULTITEXCOORDP4UIPROC __glewMultiTexCoordP4ui = NULL;
PFNGLMULTITEXCOORDP4UIVPROC __glewMultiTexCoordP4uiv = NULL;
PFNGLNORMALP3UIPROC __glewNormalP3ui = NULL;
PFNGLNORMALP3UIVPROC __glewNormalP3uiv = NULL;
PFNGLSECONDARYCOLORP3UIPROC __glewSecondaryColorP3ui = NULL;
PFNGLSECONDARYCOLORP3UIVPROC __glewSecondaryColorP3uiv = NULL;
PFNGLTEXCOORDP1UIPROC __glewTexCoordP1ui = NULL;
PFNGLTEXCOORDP1UIVPROC __glewTexCoordP1uiv = NULL;
PFNGLTEXCOORDP2UIPROC __glewTexCoordP2ui = NULL;
PFNGLTEXCOORDP2UIVPROC __glewTexCoordP2uiv = NULL;
PFNGLTEXCOORDP3UIPROC __glewTexCoordP3ui = NULL;
PFNGLTEXCOORDP3UIVPROC __glewTexCoordP3uiv = NULL;
PFNGLTEXCOORDP4UIPROC __glewTexCoordP4ui = NULL;
PFNGLTEXCOORDP4UIVPROC __glewTexCoordP4uiv = NULL;
PFNGLVERTEXATTRIBP1UIPROC __glewVertexAttribP1ui = NULL;
PFNGLVERTEXATTRIBP1UIVPROC __glewVertexAttribP1uiv = NULL;
PFNGLVERTEXATTRIBP2UIPROC __glewVertexAttribP2ui = NULL;
PFNGLVERTEXATTRIBP2UIVPROC __glewVertexAttribP2uiv = NULL;
PFNGLVERTEXATTRIBP3UIPROC __glewVertexAttribP3ui = NULL;
PFNGLVERTEXATTRIBP3UIVPROC __glewVertexAttribP3uiv = NULL;
PFNGLVERTEXATTRIBP4UIPROC __glewVertexAttribP4ui = NULL;
PFNGLVERTEXATTRIBP4UIVPROC __glewVertexAttribP4uiv = NULL;
PFNGLVERTEXP2UIPROC __glewVertexP2ui = NULL;
PFNGLVERTEXP2UIVPROC __glewVertexP2uiv = NULL;
PFNGLVERTEXP3UIPROC __glewVertexP3ui = NULL;
PFNGLVERTEXP3UIVPROC __glewVertexP3uiv = NULL;
PFNGLVERTEXP4UIPROC __glewVertexP4ui = NULL;
PFNGLVERTEXP4UIVPROC __glewVertexP4uiv = NULL;

PFNGLDEPTHRANGEARRAYVPROC __glewDepthRangeArrayv = NULL;
PFNGLDEPTHRANGEINDEXEDPROC __glewDepthRangeIndexed = NULL;
PFNGLGETDOUBLEI_VPROC __glewGetDoublei_v = NULL;
PFNGLGETFLOATI_VPROC __glewGetFloati_v = NULL;
PFNGLSCISSORARRAYVPROC __glewScissorArrayv = NULL;
PFNGLSCISSORINDEXEDPROC __glewScissorIndexed = NULL;
PFNGLSCISSORINDEXEDVPROC __glewScissorIndexedv = NULL;
PFNGLVIEWPORTARRAYVPROC __glewViewportArrayv = NULL;
PFNGLVIEWPORTINDEXEDFPROC __glewViewportIndexedf = NULL;
PFNGLVIEWPORTINDEXEDFVPROC __glewViewportIndexedfv = NULL;

PFNGLWINDOWPOS2DARBPROC __glewWindowPos2dARB = NULL;
PFNGLWINDOWPOS2DVARBPROC __glewWindowPos2dvARB = NULL;
PFNGLWINDOWPOS2FARBPROC __glewWindowPos2fARB = NULL;
PFNGLWINDOWPOS2FVARBPROC __glewWindowPos2fvARB = NULL;
PFNGLWINDOWPOS2IARBPROC __glewWindowPos2iARB = NULL;
PFNGLWINDOWPOS2IVARBPROC __glewWindowPos2ivARB = NULL;
PFNGLWINDOWPOS2SARBPROC __glewWindowPos2sARB = NULL;
PFNGLWINDOWPOS2SVARBPROC __glewWindowPos2svARB = NULL;
PFNGLWINDOWPOS3DARBPROC __glewWindowPos3dARB = NULL;
PFNGLWINDOWPOS3DVARBPROC __glewWindowPos3dvARB = NULL;
PFNGLWINDOWPOS3FARBPROC __glewWindowPos3fARB = NULL;
PFNGLWINDOWPOS3FVARBPROC __glewWindowPos3fvARB = NULL;
PFNGLWINDOWPOS3IARBPROC __glewWindowPos3iARB = NULL;
PFNGLWINDOWPOS3IVARBPROC __glewWindowPos3ivARB = NULL;
PFNGLWINDOWPOS3SARBPROC __glewWindowPos3sARB = NULL;
PFNGLWINDOWPOS3SVARBPROC __glewWindowPos3svARB = NULL;

PFNGLDRAWBUFFERSATIPROC __glewDrawBuffersATI = NULL;

PFNGLDRAWELEMENTARRAYATIPROC __glewDrawElementArrayATI = NULL;
PFNGLDRAWRANGEELEMENTARRAYATIPROC __glewDrawRangeElementArrayATI = NULL;
PFNGLELEMENTPOINTERATIPROC __glewElementPointerATI = NULL;

PFNGLGETTEXBUMPPARAMETERFVATIPROC __glewGetTexBumpParameterfvATI = NULL;
PFNGLGETTEXBUMPPARAMETERIVATIPROC __glewGetTexBumpParameterivATI = NULL;
PFNGLTEXBUMPPARAMETERFVATIPROC __glewTexBumpParameterfvATI = NULL;
PFNGLTEXBUMPPARAMETERIVATIPROC __glewTexBumpParameterivATI = NULL;

PFNGLALPHAFRAGMENTOP1ATIPROC __glewAlphaFragmentOp1ATI = NULL;
PFNGLALPHAFRAGMENTOP2ATIPROC __glewAlphaFragmentOp2ATI = NULL;
PFNGLALPHAFRAGMENTOP3ATIPROC __glewAlphaFragmentOp3ATI = NULL;
PFNGLBEGINFRAGMENTSHADERATIPROC __glewBeginFragmentShaderATI = NULL;
PFNGLBINDFRAGMENTSHADERATIPROC __glewBindFragmentShaderATI = NULL;
PFNGLCOLORFRAGMENTOP1ATIPROC __glewColorFragmentOp1ATI = NULL;
PFNGLCOLORFRAGMENTOP2ATIPROC __glewColorFragmentOp2ATI = NULL;
PFNGLCOLORFRAGMENTOP3ATIPROC __glewColorFragmentOp3ATI = NULL;
PFNGLDELETEFRAGMENTSHADERATIPROC __glewDeleteFragmentShaderATI = NULL;
PFNGLENDFRAGMENTSHADERATIPROC __glewEndFragmentShaderATI = NULL;
PFNGLGENFRAGMENTSHADERSATIPROC __glewGenFragmentShadersATI = NULL;
PFNGLPASSTEXCOORDATIPROC __glewPassTexCoordATI = NULL;
PFNGLSAMPLEMAPATIPROC __glewSampleMapATI = NULL;
PFNGLSETFRAGMENTSHADERCONSTANTATIPROC __glewSetFragmentShaderConstantATI = NULL;

PFNGLMAPOBJECTBUFFERATIPROC __glewMapObjectBufferATI = NULL;
PFNGLUNMAPOBJECTBUFFERATIPROC __glewUnmapObjectBufferATI = NULL;

PFNGLPNTRIANGLESFATIPROC __glewPNTrianglesfATI = NULL;
PFNGLPNTRIANGLESIATIPROC __glewPNTrianglesiATI = NULL;

PFNGLSTENCILFUNCSEPARATEATIPROC __glewStencilFuncSeparateATI = NULL;
PFNGLSTENCILOPSEPARATEATIPROC __glewStencilOpSeparateATI = NULL;

PFNGLARRAYOBJECTATIPROC __glewArrayObjectATI = NULL;
PFNGLFREEOBJECTBUFFERATIPROC __glewFreeObjectBufferATI = NULL;
PFNGLGETARRAYOBJECTFVATIPROC __glewGetArrayObjectfvATI = NULL;
PFNGLGETARRAYOBJECTIVATIPROC __glewGetArrayObjectivATI = NULL;
PFNGLGETOBJECTBUFFERFVATIPROC __glewGetObjectBufferfvATI = NULL;
PFNGLGETOBJECTBUFFERIVATIPROC __glewGetObjectBufferivATI = NULL;
PFNGLGETVARIANTARRAYOBJECTFVATIPROC __glewGetVariantArrayObjectfvATI = NULL;
PFNGLGETVARIANTARRAYOBJECTIVATIPROC __glewGetVariantArrayObjectivATI = NULL;
PFNGLISOBJECTBUFFERATIPROC __glewIsObjectBufferATI = NULL;
PFNGLNEWOBJECTBUFFERATIPROC __glewNewObjectBufferATI = NULL;
PFNGLUPDATEOBJECTBUFFERATIPROC __glewUpdateObjectBufferATI = NULL;
PFNGLVARIANTARRAYOBJECTATIPROC __glewVariantArrayObjectATI = NULL;

PFNGLGETVERTEXATTRIBARRAYOBJECTFVATIPROC __glewGetVertexAttribArrayObjectfvATI = NULL;
PFNGLGETVERTEXATTRIBARRAYOBJECTIVATIPROC __glewGetVertexAttribArrayObjectivATI = NULL;
PFNGLVERTEXATTRIBARRAYOBJECTATIPROC __glewVertexAttribArrayObjectATI = NULL;

PFNGLCLIENTACTIVEVERTEXSTREAMATIPROC __glewClientActiveVertexStreamATI = NULL;
PFNGLNORMALSTREAM3BATIPROC __glewNormalStream3bATI = NULL;
PFNGLNORMALSTREAM3BVATIPROC __glewNormalStream3bvATI = NULL;
PFNGLNORMALSTREAM3DATIPROC __glewNormalStream3dATI = NULL;
PFNGLNORMALSTREAM3DVATIPROC __glewNormalStream3dvATI = NULL;
PFNGLNORMALSTREAM3FATIPROC __glewNormalStream3fATI = NULL;
PFNGLNORMALSTREAM3FVATIPROC __glewNormalStream3fvATI = NULL;
PFNGLNORMALSTREAM3IATIPROC __glewNormalStream3iATI = NULL;
PFNGLNORMALSTREAM3IVATIPROC __glewNormalStream3ivATI = NULL;
PFNGLNORMALSTREAM3SATIPROC __glewNormalStream3sATI = NULL;
PFNGLNORMALSTREAM3SVATIPROC __glewNormalStream3svATI = NULL;
PFNGLVERTEXBLENDENVFATIPROC __glewVertexBlendEnvfATI = NULL;
PFNGLVERTEXBLENDENVIATIPROC __glewVertexBlendEnviATI = NULL;
PFNGLVERTEXSTREAM1DATIPROC __glewVertexStream1dATI = NULL;
PFNGLVERTEXSTREAM1DVATIPROC __glewVertexStream1dvATI = NULL;
PFNGLVERTEXSTREAM1FATIPROC __glewVertexStream1fATI = NULL;
PFNGLVERTEXSTREAM1FVATIPROC __glewVertexStream1fvATI = NULL;
PFNGLVERTEXSTREAM1IATIPROC __glewVertexStream1iATI = NULL;
PFNGLVERTEXSTREAM1IVATIPROC __glewVertexStream1ivATI = NULL;
PFNGLVERTEXSTREAM1SATIPROC __glewVertexStream1sATI = NULL;
PFNGLVERTEXSTREAM1SVATIPROC __glewVertexStream1svATI = NULL;
PFNGLVERTEXSTREAM2DATIPROC __glewVertexStream2dATI = NULL;
PFNGLVERTEXSTREAM2DVATIPROC __glewVertexStream2dvATI = NULL;
PFNGLVERTEXSTREAM2FATIPROC __glewVertexStream2fATI = NULL;
PFNGLVERTEXSTREAM2FVATIPROC __glewVertexStream2fvATI = NULL;
PFNGLVERTEXSTREAM2IATIPROC __glewVertexStream2iATI = NULL;
PFNGLVERTEXSTREAM2IVATIPROC __glewVertexStream2ivATI = NULL;
PFNGLVERTEXSTREAM2SATIPROC __glewVertexStream2sATI = NULL;
PFNGLVERTEXSTREAM2SVATIPROC __glewVertexStream2svATI = NULL;
PFNGLVERTEXSTREAM3DATIPROC __glewVertexStream3dATI = NULL;
PFNGLVERTEXSTREAM3DVATIPROC __glewVertexStream3dvATI = NULL;
PFNGLVERTEXSTREAM3FATIPROC __glewVertexStream3fATI = NULL;
PFNGLVERTEXSTREAM3FVATIPROC __glewVertexStream3fvATI = NULL;
PFNGLVERTEXSTREAM3IATIPROC __glewVertexStream3iATI = NULL;
PFNGLVERTEXSTREAM3IVATIPROC __glewVertexStream3ivATI = NULL;
PFNGLVERTEXSTREAM3SATIPROC __glewVertexStream3sATI = NULL;
PFNGLVERTEXSTREAM3SVATIPROC __glewVertexStream3svATI = NULL;
PFNGLVERTEXSTREAM4DATIPROC __glewVertexStream4dATI = NULL;
PFNGLVERTEXSTREAM4DVATIPROC __glewVertexStream4dvATI = NULL;
PFNGLVERTEXSTREAM4FATIPROC __glewVertexStream4fATI = NULL;
PFNGLVERTEXSTREAM4FVATIPROC __glewVertexStream4fvATI = NULL;
PFNGLVERTEXSTREAM4IATIPROC __glewVertexStream4iATI = NULL;
PFNGLVERTEXSTREAM4IVATIPROC __glewVertexStream4ivATI = NULL;
PFNGLVERTEXSTREAM4SATIPROC __glewVertexStream4sATI = NULL;
PFNGLVERTEXSTREAM4SVATIPROC __glewVertexStream4svATI = NULL;

PFNGLGETUNIFORMBUFFERSIZEEXTPROC __glewGetUniformBufferSizeEXT = NULL;
PFNGLGETUNIFORMOFFSETEXTPROC __glewGetUniformOffsetEXT = NULL;
PFNGLUNIFORMBUFFEREXTPROC __glewUniformBufferEXT = NULL;

PFNGLBLENDCOLOREXTPROC __glewBlendColorEXT = NULL;

PFNGLBLENDEQUATIONSEPARATEEXTPROC __glewBlendEquationSeparateEXT = NULL;

PFNGLBLENDFUNCSEPARATEEXTPROC __glewBlendFuncSeparateEXT = NULL;

PFNGLBLENDEQUATIONEXTPROC __glewBlendEquationEXT = NULL;

PFNGLCOLORSUBTABLEEXTPROC __glewColorSubTableEXT = NULL;
PFNGLCOPYCOLORSUBTABLEEXTPROC __glewCopyColorSubTableEXT = NULL;

PFNGLLOCKARRAYSEXTPROC __glewLockArraysEXT = NULL;
PFNGLUNLOCKARRAYSEXTPROC __glewUnlockArraysEXT = NULL;

PFNGLCONVOLUTIONFILTER1DEXTPROC __glewConvolutionFilter1DEXT = NULL;
PFNGLCONVOLUTIONFILTER2DEXTPROC __glewConvolutionFilter2DEXT = NULL;
PFNGLCONVOLUTIONPARAMETERFEXTPROC __glewConvolutionParameterfEXT = NULL;
PFNGLCONVOLUTIONPARAMETERFVEXTPROC __glewConvolutionParameterfvEXT = NULL;
PFNGLCONVOLUTIONPARAMETERIEXTPROC __glewConvolutionParameteriEXT = NULL;
PFNGLCONVOLUTIONPARAMETERIVEXTPROC __glewConvolutionParameterivEXT = NULL;
PFNGLCOPYCONVOLUTIONFILTER1DEXTPROC __glewCopyConvolutionFilter1DEXT = NULL;
PFNGLCOPYCONVOLUTIONFILTER2DEXTPROC __glewCopyConvolutionFilter2DEXT = NULL;
PFNGLGETCONVOLUTIONFILTEREXTPROC __glewGetConvolutionFilterEXT = NULL;
PFNGLGETCONVOLUTIONPARAMETERFVEXTPROC __glewGetConvolutionParameterfvEXT = NULL;
PFNGLGETCONVOLUTIONPARAMETERIVEXTPROC __glewGetConvolutionParameterivEXT = NULL;
PFNGLGETSEPARABLEFILTEREXTPROC __glewGetSeparableFilterEXT = NULL;
PFNGLSEPARABLEFILTER2DEXTPROC __glewSeparableFilter2DEXT = NULL;

PFNGLBINORMALPOINTEREXTPROC __glewBinormalPointerEXT = NULL;
PFNGLTANGENTPOINTEREXTPROC __glewTangentPointerEXT = NULL;

PFNGLCOPYTEXIMAGE1DEXTPROC __glewCopyTexImage1DEXT = NULL;
PFNGLCOPYTEXIMAGE2DEXTPROC __glewCopyTexImage2DEXT = NULL;
PFNGLCOPYTEXSUBIMAGE1DEXTPROC __glewCopyTexSubImage1DEXT = NULL;
PFNGLCOPYTEXSUBIMAGE2DEXTPROC __glewCopyTexSubImage2DEXT = NULL;
PFNGLCOPYTEXSUBIMAGE3DEXTPROC __glewCopyTexSubImage3DEXT = NULL;

PFNGLCULLPARAMETERDVEXTPROC __glewCullParameterdvEXT = NULL;
PFNGLCULLPARAMETERFVEXTPROC __glewCullParameterfvEXT = NULL;

PFNGLGETOBJECTLABELEXTPROC __glewGetObjectLabelEXT = NULL;
PFNGLLABELOBJECTEXTPROC __glewLabelObjectEXT = NULL;

PFNGLINSERTEVENTMARKEREXTPROC __glewInsertEventMarkerEXT = NULL;
PFNGLPOPGROUPMARKEREXTPROC __glewPopGroupMarkerEXT = NULL;
PFNGLPUSHGROUPMARKEREXTPROC __glewPushGroupMarkerEXT = NULL;

PFNGLDEPTHBOUNDSEXTPROC __glewDepthBoundsEXT = NULL;

PFNGLBINDMULTITEXTUREEXTPROC __glewBindMultiTextureEXT = NULL;
PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC __glewCheckNamedFramebufferStatusEXT = NULL;
PFNGLCLIENTATTRIBDEFAULTEXTPROC __glewClientAttribDefaultEXT = NULL;
PFNGLCOMPRESSEDMULTITEXIMAGE1DEXTPROC __glewCompressedMultiTexImage1DEXT = NULL;
PFNGLCOMPRESSEDMULTITEXIMAGE2DEXTPROC __glewCompressedMultiTexImage2DEXT = NULL;
PFNGLCOMPRESSEDMULTITEXIMAGE3DEXTPROC __glewCompressedMultiTexImage3DEXT = NULL;
PFNGLCOMPRESSEDMULTITEXSUBIMAGE1DEXTPROC __glewCompressedMultiTexSubImage1DEXT = NULL;
PFNGLCOMPRESSEDMULTITEXSUBIMAGE2DEXTPROC __glewCompressedMultiTexSubImage2DEXT = NULL;
PFNGLCOMPRESSEDMULTITEXSUBIMAGE3DEXTPROC __glewCompressedMultiTexSubImage3DEXT = NULL;
PFNGLCOMPRESSEDTEXTUREIMAGE1DEXTPROC __glewCompressedTextureImage1DEXT = NULL;
PFNGLCOMPRESSEDTEXTUREIMAGE2DEXTPROC __glewCompressedTextureImage2DEXT = NULL;
PFNGLCOMPRESSEDTEXTUREIMAGE3DEXTPROC __glewCompressedTextureImage3DEXT = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC __glewCompressedTextureSubImage1DEXT = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC __glewCompressedTextureSubImage2DEXT = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC __glewCompressedTextureSubImage3DEXT = NULL;
PFNGLCOPYMULTITEXIMAGE1DEXTPROC __glewCopyMultiTexImage1DEXT = NULL;
PFNGLCOPYMULTITEXIMAGE2DEXTPROC __glewCopyMultiTexImage2DEXT = NULL;
PFNGLCOPYMULTITEXSUBIMAGE1DEXTPROC __glewCopyMultiTexSubImage1DEXT = NULL;
PFNGLCOPYMULTITEXSUBIMAGE2DEXTPROC __glewCopyMultiTexSubImage2DEXT = NULL;
PFNGLCOPYMULTITEXSUBIMAGE3DEXTPROC __glewCopyMultiTexSubImage3DEXT = NULL;
PFNGLCOPYTEXTUREIMAGE1DEXTPROC __glewCopyTextureImage1DEXT = NULL;
PFNGLCOPYTEXTUREIMAGE2DEXTPROC __glewCopyTextureImage2DEXT = NULL;
PFNGLCOPYTEXTURESUBIMAGE1DEXTPROC __glewCopyTextureSubImage1DEXT = NULL;
PFNGLCOPYTEXTURESUBIMAGE2DEXTPROC __glewCopyTextureSubImage2DEXT = NULL;
PFNGLCOPYTEXTURESUBIMAGE3DEXTPROC __glewCopyTextureSubImage3DEXT = NULL;
PFNGLDISABLECLIENTSTATEINDEXEDEXTPROC __glewDisableClientStateIndexedEXT = NULL;
PFNGLDISABLECLIENTSTATEIEXTPROC __glewDisableClientStateiEXT = NULL;
PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC __glewDisableVertexArrayAttribEXT = NULL;
PFNGLDISABLEVERTEXARRAYEXTPROC __glewDisableVertexArrayEXT = NULL;
PFNGLENABLECLIENTSTATEINDEXEDEXTPROC __glewEnableClientStateIndexedEXT = NULL;
PFNGLENABLECLIENTSTATEIEXTPROC __glewEnableClientStateiEXT = NULL;
PFNGLENABLEVERTEXARRAYATTRIBEXTPROC __glewEnableVertexArrayAttribEXT = NULL;
PFNGLENABLEVERTEXARRAYEXTPROC __glewEnableVertexArrayEXT = NULL;
PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC __glewFlushMappedNamedBufferRangeEXT = NULL;
PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC __glewFramebufferDrawBufferEXT = NULL;
PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC __glewFramebufferDrawBuffersEXT = NULL;
PFNGLFRAMEBUFFERREADBUFFEREXTPROC __glewFramebufferReadBufferEXT = NULL;
PFNGLGENERATEMULTITEXMIPMAPEXTPROC __glewGenerateMultiTexMipmapEXT = NULL;
PFNGLGENERATETEXTUREMIPMAPEXTPROC __glewGenerateTextureMipmapEXT = NULL;
PFNGLGETCOMPRESSEDMULTITEXIMAGEEXTPROC __glewGetCompressedMultiTexImageEXT = NULL;
PFNGLGETCOMPRESSEDTEXTUREIMAGEEXTPROC __glewGetCompressedTextureImageEXT = NULL;
PFNGLGETDOUBLEINDEXEDVEXTPROC __glewGetDoubleIndexedvEXT = NULL;
PFNGLGETDOUBLEI_VEXTPROC __glewGetDoublei_vEXT = NULL;
PFNGLGETFLOATINDEXEDVEXTPROC __glewGetFloatIndexedvEXT = NULL;
PFNGLGETFLOATI_VEXTPROC __glewGetFloati_vEXT = NULL;
PFNGLGETFRAMEBUFFERPARAMETERIVEXTPROC __glewGetFramebufferParameterivEXT = NULL;
PFNGLGETMULTITEXENVFVEXTPROC __glewGetMultiTexEnvfvEXT = NULL;
PFNGLGETMULTITEXENVIVEXTPROC __glewGetMultiTexEnvivEXT = NULL;
PFNGLGETMULTITEXGENDVEXTPROC __glewGetMultiTexGendvEXT = NULL;
PFNGLGETMULTITEXGENFVEXTPROC __glewGetMultiTexGenfvEXT = NULL;
PFNGLGETMULTITEXGENIVEXTPROC __glewGetMultiTexGenivEXT = NULL;
PFNGLGETMULTITEXIMAGEEXTPROC __glewGetMultiTexImageEXT = NULL;
PFNGLGETMULTITEXLEVELPARAMETERFVEXTPROC __glewGetMultiTexLevelParameterfvEXT = NULL;
PFNGLGETMULTITEXLEVELPARAMETERIVEXTPROC __glewGetMultiTexLevelParameterivEXT = NULL;
PFNGLGETMULTITEXPARAMETERIIVEXTPROC __glewGetMultiTexParameterIivEXT = NULL;
PFNGLGETMULTITEXPARAMETERIUIVEXTPROC __glewGetMultiTexParameterIuivEXT = NULL;
PFNGLGETMULTITEXPARAMETERFVEXTPROC __glewGetMultiTexParameterfvEXT = NULL;
PFNGLGETMULTITEXPARAMETERIVEXTPROC __glewGetMultiTexParameterivEXT = NULL;
PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC __glewGetNamedBufferParameterivEXT = NULL;
PFNGLGETNAMEDBUFFERPOINTERVEXTPROC __glewGetNamedBufferPointervEXT = NULL;
PFNGLGETNAMEDBUFFERSUBDATAEXTPROC __glewGetNamedBufferSubDataEXT = NULL;
PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC __glewGetNamedFramebufferAttachmentParameterivEXT = NULL;
PFNGLGETNAMEDPROGRAMLOCALPARAMETERIIVEXTPROC __glewGetNamedProgramLocalParameterIivEXT = NULL;
PFNGLGETNAMEDPROGRAMLOCALPARAMETERIUIVEXTPROC __glewGetNamedProgramLocalParameterIuivEXT = NULL;
PFNGLGETNAMEDPROGRAMLOCALPARAMETERDVEXTPROC __glewGetNamedProgramLocalParameterdvEXT = NULL;
PFNGLGETNAMEDPROGRAMLOCALPARAMETERFVEXTPROC __glewGetNamedProgramLocalParameterfvEXT = NULL;
PFNGLGETNAMEDPROGRAMSTRINGEXTPROC __glewGetNamedProgramStringEXT = NULL;
PFNGLGETNAMEDPROGRAMIVEXTPROC __glewGetNamedProgramivEXT = NULL;
PFNGLGETNAMEDRENDERBUFFERPARAMETERIVEXTPROC __glewGetNamedRenderbufferParameterivEXT = NULL;
PFNGLGETPOINTERINDEXEDVEXTPROC __glewGetPointerIndexedvEXT = NULL;
PFNGLGETPOINTERI_VEXTPROC __glewGetPointeri_vEXT = NULL;
PFNGLGETTEXTUREIMAGEEXTPROC __glewGetTextureImageEXT = NULL;
PFNGLGETTEXTURELEVELPARAMETERFVEXTPROC __glewGetTextureLevelParameterfvEXT = NULL;
PFNGLGETTEXTURELEVELPARAMETERIVEXTPROC __glewGetTextureLevelParameterivEXT = NULL;
PFNGLGETTEXTUREPARAMETERIIVEXTPROC __glewGetTextureParameterIivEXT = NULL;
PFNGLGETTEXTUREPARAMETERIUIVEXTPROC __glewGetTextureParameterIuivEXT = NULL;
PFNGLGETTEXTUREPARAMETERFVEXTPROC __glewGetTextureParameterfvEXT = NULL;
PFNGLGETTEXTUREPARAMETERIVEXTPROC __glewGetTextureParameterivEXT = NULL;
PFNGLGETVERTEXARRAYINTEGERI_VEXTPROC __glewGetVertexArrayIntegeri_vEXT = NULL;
PFNGLGETVERTEXARRAYINTEGERVEXTPROC __glewGetVertexArrayIntegervEXT = NULL;
PFNGLGETVERTEXARRAYPOINTERI_VEXTPROC __glewGetVertexArrayPointeri_vEXT = NULL;
PFNGLGETVERTEXARRAYPOINTERVEXTPROC __glewGetVertexArrayPointervEXT = NULL;
PFNGLMAPNAMEDBUFFEREXTPROC __glewMapNamedBufferEXT = NULL;
PFNGLMAPNAMEDBUFFERRANGEEXTPROC __glewMapNamedBufferRangeEXT = NULL;
PFNGLMATRIXFRUSTUMEXTPROC __glewMatrixFrustumEXT = NULL;
PFNGLMATRIXLOADIDENTITYEXTPROC __glewMatrixLoadIdentityEXT = NULL;
PFNGLMATRIXLOADTRANSPOSEDEXTPROC __glewMatrixLoadTransposedEXT = NULL;
PFNGLMATRIXLOADTRANSPOSEFEXTPROC __glewMatrixLoadTransposefEXT = NULL;
PFNGLMATRIXLOADDEXTPROC __glewMatrixLoaddEXT = NULL;
PFNGLMATRIXLOADFEXTPROC __glewMatrixLoadfEXT = NULL;
PFNGLMATRIXMULTTRANSPOSEDEXTPROC __glewMatrixMultTransposedEXT = NULL;
PFNGLMATRIXMULTTRANSPOSEFEXTPROC __glewMatrixMultTransposefEXT = NULL;
PFNGLMATRIXMULTDEXTPROC __glewMatrixMultdEXT = NULL;
PFNGLMATRIXMULTFEXTPROC __glewMatrixMultfEXT = NULL;
PFNGLMATRIXORTHOEXTPROC __glewMatrixOrthoEXT = NULL;
PFNGLMATRIXPOPEXTPROC __glewMatrixPopEXT = NULL;
PFNGLMATRIXPUSHEXTPROC __glewMatrixPushEXT = NULL;
PFNGLMATRIXROTATEDEXTPROC __glewMatrixRotatedEXT = NULL;
PFNGLMATRIXROTATEFEXTPROC __glewMatrixRotatefEXT = NULL;
PFNGLMATRIXSCALEDEXTPROC __glewMatrixScaledEXT = NULL;
PFNGLMATRIXSCALEFEXTPROC __glewMatrixScalefEXT = NULL;
PFNGLMATRIXTRANSLATEDEXTPROC __glewMatrixTranslatedEXT = NULL;
PFNGLMATRIXTRANSLATEFEXTPROC __glewMatrixTranslatefEXT = NULL;
PFNGLMULTITEXBUFFEREXTPROC __glewMultiTexBufferEXT = NULL;
PFNGLMULTITEXCOORDPOINTEREXTPROC __glewMultiTexCoordPointerEXT = NULL;
PFNGLMULTITEXENVFEXTPROC __glewMultiTexEnvfEXT = NULL;
PFNGLMULTITEXENVFVEXTPROC __glewMultiTexEnvfvEXT = NULL;
PFNGLMULTITEXENVIEXTPROC __glewMultiTexEnviEXT = NULL;
PFNGLMULTITEXENVIVEXTPROC __glewMultiTexEnvivEXT = NULL;
PFNGLMULTITEXGENDEXTPROC __glewMultiTexGendEXT = NULL;
PFNGLMULTITEXGENDVEXTPROC __glewMultiTexGendvEXT = NULL;
PFNGLMULTITEXGENFEXTPROC __glewMultiTexGenfEXT = NULL;
PFNGLMULTITEXGENFVEXTPROC __glewMultiTexGenfvEXT = NULL;
PFNGLMULTITEXGENIEXTPROC __glewMultiTexGeniEXT = NULL;
PFNGLMULTITEXGENIVEXTPROC __glewMultiTexGenivEXT = NULL;
PFNGLMULTITEXIMAGE1DEXTPROC __glewMultiTexImage1DEXT = NULL;
PFNGLMULTITEXIMAGE2DEXTPROC __glewMultiTexImage2DEXT = NULL;
PFNGLMULTITEXIMAGE3DEXTPROC __glewMultiTexImage3DEXT = NULL;
PFNGLMULTITEXPARAMETERIIVEXTPROC __glewMultiTexParameterIivEXT = NULL;
PFNGLMULTITEXPARAMETERIUIVEXTPROC __glewMultiTexParameterIuivEXT = NULL;
PFNGLMULTITEXPARAMETERFEXTPROC __glewMultiTexParameterfEXT = NULL;
PFNGLMULTITEXPARAMETERFVEXTPROC __glewMultiTexParameterfvEXT = NULL;
PFNGLMULTITEXPARAMETERIEXTPROC __glewMultiTexParameteriEXT = NULL;
PFNGLMULTITEXPARAMETERIVEXTPROC __glewMultiTexParameterivEXT = NULL;
PFNGLMULTITEXRENDERBUFFEREXTPROC __glewMultiTexRenderbufferEXT = NULL;
PFNGLMULTITEXSUBIMAGE1DEXTPROC __glewMultiTexSubImage1DEXT = NULL;
PFNGLMULTITEXSUBIMAGE2DEXTPROC __glewMultiTexSubImage2DEXT = NULL;
PFNGLMULTITEXSUBIMAGE3DEXTPROC __glewMultiTexSubImage3DEXT = NULL;
PFNGLNAMEDBUFFERDATAEXTPROC __glewNamedBufferDataEXT = NULL;
PFNGLNAMEDBUFFERSUBDATAEXTPROC __glewNamedBufferSubDataEXT = NULL;
PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC __glewNamedCopyBufferSubDataEXT = NULL;
PFNGLNAMEDFRAMEBUFFERRENDERBUFFEREXTPROC __glewNamedFramebufferRenderbufferEXT = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTURE1DEXTPROC __glewNamedFramebufferTexture1DEXT = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTURE2DEXTPROC __glewNamedFramebufferTexture2DEXT = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTURE3DEXTPROC __glewNamedFramebufferTexture3DEXT = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC __glewNamedFramebufferTextureEXT = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTUREFACEEXTPROC __glewNamedFramebufferTextureFaceEXT = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC __glewNamedFramebufferTextureLayerEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETER4DEXTPROC __glewNamedProgramLocalParameter4dEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETER4DVEXTPROC __glewNamedProgramLocalParameter4dvEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETER4FEXTPROC __glewNamedProgramLocalParameter4fEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETER4FVEXTPROC __glewNamedProgramLocalParameter4fvEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETERI4IEXTPROC __glewNamedProgramLocalParameterI4iEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETERI4IVEXTPROC __glewNamedProgramLocalParameterI4ivEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETERI4UIEXTPROC __glewNamedProgramLocalParameterI4uiEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETERI4UIVEXTPROC __glewNamedProgramLocalParameterI4uivEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETERS4FVEXTPROC __glewNamedProgramLocalParameters4fvEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETERSI4IVEXTPROC __glewNamedProgramLocalParametersI4ivEXT = NULL;
PFNGLNAMEDPROGRAMLOCALPARAMETERSI4UIVEXTPROC __glewNamedProgramLocalParametersI4uivEXT = NULL;
PFNGLNAMEDPROGRAMSTRINGEXTPROC __glewNamedProgramStringEXT = NULL;
PFNGLNAMEDRENDERBUFFERSTORAGEEXTPROC __glewNamedRenderbufferStorageEXT = NULL;
PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLECOVERAGEEXTPROC __glewNamedRenderbufferStorageMultisampleCoverageEXT = NULL;
PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC __glewNamedRenderbufferStorageMultisampleEXT = NULL;
PFNGLPROGRAMUNIFORM1FEXTPROC __glewProgramUniform1fEXT = NULL;
PFNGLPROGRAMUNIFORM1FVEXTPROC __glewProgramUniform1fvEXT = NULL;
PFNGLPROGRAMUNIFORM1IEXTPROC __glewProgramUniform1iEXT = NULL;
PFNGLPROGRAMUNIFORM1IVEXTPROC __glewProgramUniform1ivEXT = NULL;
PFNGLPROGRAMUNIFORM1UIEXTPROC __glewProgramUniform1uiEXT = NULL;
PFNGLPROGRAMUNIFORM1UIVEXTPROC __glewProgramUniform1uivEXT = NULL;
PFNGLPROGRAMUNIFORM2FEXTPROC __glewProgramUniform2fEXT = NULL;
PFNGLPROGRAMUNIFORM2FVEXTPROC __glewProgramUniform2fvEXT = NULL;
PFNGLPROGRAMUNIFORM2IEXTPROC __glewProgramUniform2iEXT = NULL;
PFNGLPROGRAMUNIFORM2IVEXTPROC __glewProgramUniform2ivEXT = NULL;
PFNGLPROGRAMUNIFORM2UIEXTPROC __glewProgramUniform2uiEXT = NULL;
PFNGLPROGRAMUNIFORM2UIVEXTPROC __glewProgramUniform2uivEXT = NULL;
PFNGLPROGRAMUNIFORM3FEXTPROC __glewProgramUniform3fEXT = NULL;
PFNGLPROGRAMUNIFORM3FVEXTPROC __glewProgramUniform3fvEXT = NULL;
PFNGLPROGRAMUNIFORM3IEXTPROC __glewProgramUniform3iEXT = NULL;
PFNGLPROGRAMUNIFORM3IVEXTPROC __glewProgramUniform3ivEXT = NULL;
PFNGLPROGRAMUNIFORM3UIEXTPROC __glewProgramUniform3uiEXT = NULL;
PFNGLPROGRAMUNIFORM3UIVEXTPROC __glewProgramUniform3uivEXT = NULL;
PFNGLPROGRAMUNIFORM4FEXTPROC __glewProgramUniform4fEXT = NULL;
PFNGLPROGRAMUNIFORM4FVEXTPROC __glewProgramUniform4fvEXT = NULL;
PFNGLPROGRAMUNIFORM4IEXTPROC __glewProgramUniform4iEXT = NULL;
PFNGLPROGRAMUNIFORM4IVEXTPROC __glewProgramUniform4ivEXT = NULL;
PFNGLPROGRAMUNIFORM4UIEXTPROC __glewProgramUniform4uiEXT = NULL;
PFNGLPROGRAMUNIFORM4UIVEXTPROC __glewProgramUniform4uivEXT = NULL;
PFNGLPROGRAMUNIFORMMATRIX2FVEXTPROC __glewProgramUniformMatrix2fvEXT = NULL;
PFNGLPROGRAMUNIFORMMATRIX2X3FVEXTPROC __glewProgramUniformMatrix2x3fvEXT = NULL;
PFNGLPROGRAMUNIFORMMATRIX2X4FVEXTPROC __glewProgramUniformMatrix2x4fvEXT = NULL;
PFNGLPROGRAMUNIFORMMATRIX3FVEXTPROC __glewProgramUniformMatrix3fvEXT = NULL;
PFNGLPROGRAMUNIFORMMATRIX3X2FVEXTPROC __glewProgramUniformMatrix3x2fvEXT = NULL;
PFNGLPROGRAMUNIFORMMATRIX3X4FVEXTPROC __glewProgramUniformMatrix3x4fvEXT = NULL;
PFNGLPROGRAMUNIFORMMATRIX4FVEXTPROC __glewProgramUniformMatrix4fvEXT = NULL;
PFNGLPROGRAMUNIFORMMATRIX4X2FVEXTPROC __glewProgramUniformMatrix4x2fvEXT = NULL;
PFNGLPROGRAMUNIFORMMATRIX4X3FVEXTPROC __glewProgramUniformMatrix4x3fvEXT = NULL;
PFNGLPUSHCLIENTATTRIBDEFAULTEXTPROC __glewPushClientAttribDefaultEXT = NULL;
PFNGLTEXTUREBUFFEREXTPROC __glewTextureBufferEXT = NULL;
PFNGLTEXTUREIMAGE1DEXTPROC __glewTextureImage1DEXT = NULL;
PFNGLTEXTUREIMAGE2DEXTPROC __glewTextureImage2DEXT = NULL;
PFNGLTEXTUREIMAGE3DEXTPROC __glewTextureImage3DEXT = NULL;
PFNGLTEXTUREPARAMETERIIVEXTPROC __glewTextureParameterIivEXT = NULL;
PFNGLTEXTUREPARAMETERIUIVEXTPROC __glewTextureParameterIuivEXT = NULL;
PFNGLTEXTUREPARAMETERFEXTPROC __glewTextureParameterfEXT = NULL;
PFNGLTEXTUREPARAMETERFVEXTPROC __glewTextureParameterfvEXT = NULL;
PFNGLTEXTUREPARAMETERIEXTPROC __glewTextureParameteriEXT = NULL;
PFNGLTEXTUREPARAMETERIVEXTPROC __glewTextureParameterivEXT = NULL;
PFNGLTEXTURERENDERBUFFEREXTPROC __glewTextureRenderbufferEXT = NULL;
PFNGLTEXTURESUBIMAGE1DEXTPROC __glewTextureSubImage1DEXT = NULL;
PFNGLTEXTURESUBIMAGE2DEXTPROC __glewTextureSubImage2DEXT = NULL;
PFNGLTEXTURESUBIMAGE3DEXTPROC __glewTextureSubImage3DEXT = NULL;
PFNGLUNMAPNAMEDBUFFEREXTPROC __glewUnmapNamedBufferEXT = NULL;
PFNGLVERTEXARRAYCOLOROFFSETEXTPROC __glewVertexArrayColorOffsetEXT = NULL;
PFNGLVERTEXARRAYEDGEFLAGOFFSETEXTPROC __glewVertexArrayEdgeFlagOffsetEXT = NULL;
PFNGLVERTEXARRAYFOGCOORDOFFSETEXTPROC __glewVertexArrayFogCoordOffsetEXT = NULL;
PFNGLVERTEXARRAYINDEXOFFSETEXTPROC __glewVertexArrayIndexOffsetEXT = NULL;
PFNGLVERTEXARRAYMULTITEXCOORDOFFSETEXTPROC __glewVertexArrayMultiTexCoordOffsetEXT = NULL;
PFNGLVERTEXARRAYNORMALOFFSETEXTPROC __glewVertexArrayNormalOffsetEXT = NULL;
PFNGLVERTEXARRAYSECONDARYCOLOROFFSETEXTPROC __glewVertexArraySecondaryColorOffsetEXT = NULL;
PFNGLVERTEXARRAYTEXCOORDOFFSETEXTPROC __glewVertexArrayTexCoordOffsetEXT = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBDIVISOREXTPROC __glewVertexArrayVertexAttribDivisorEXT = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBIOFFSETEXTPROC __glewVertexArrayVertexAttribIOffsetEXT = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBOFFSETEXTPROC __glewVertexArrayVertexAttribOffsetEXT = NULL;
PFNGLVERTEXARRAYVERTEXOFFSETEXTPROC __glewVertexArrayVertexOffsetEXT = NULL;

PFNGLCOLORMASKINDEXEDEXTPROC __glewColorMaskIndexedEXT = NULL;
PFNGLDISABLEINDEXEDEXTPROC __glewDisableIndexedEXT = NULL;
PFNGLENABLEINDEXEDEXTPROC __glewEnableIndexedEXT = NULL;
PFNGLGETBOOLEANINDEXEDVEXTPROC __glewGetBooleanIndexedvEXT = NULL;
PFNGLGETINTEGERINDEXEDVEXTPROC __glewGetIntegerIndexedvEXT = NULL;
PFNGLISENABLEDINDEXEDEXTPROC __glewIsEnabledIndexedEXT = NULL;

PFNGLDRAWARRAYSINSTANCEDEXTPROC __glewDrawArraysInstancedEXT = NULL;
PFNGLDRAWELEMENTSINSTANCEDEXTPROC __glewDrawElementsInstancedEXT = NULL;

PFNGLDRAWRANGEELEMENTSEXTPROC __glewDrawRangeElementsEXT = NULL;

PFNGLFOGCOORDPOINTEREXTPROC __glewFogCoordPointerEXT = NULL;
PFNGLFOGCOORDDEXTPROC __glewFogCoorddEXT = NULL;
PFNGLFOGCOORDDVEXTPROC __glewFogCoorddvEXT = NULL;
PFNGLFOGCOORDFEXTPROC __glewFogCoordfEXT = NULL;
PFNGLFOGCOORDFVEXTPROC __glewFogCoordfvEXT = NULL;

PFNGLFRAGMENTCOLORMATERIALEXTPROC __glewFragmentColorMaterialEXT = NULL;
PFNGLFRAGMENTLIGHTMODELFEXTPROC __glewFragmentLightModelfEXT = NULL;
PFNGLFRAGMENTLIGHTMODELFVEXTPROC __glewFragmentLightModelfvEXT = NULL;
PFNGLFRAGMENTLIGHTMODELIEXTPROC __glewFragmentLightModeliEXT = NULL;
PFNGLFRAGMENTLIGHTMODELIVEXTPROC __glewFragmentLightModelivEXT = NULL;
PFNGLFRAGMENTLIGHTFEXTPROC __glewFragmentLightfEXT = NULL;
PFNGLFRAGMENTLIGHTFVEXTPROC __glewFragmentLightfvEXT = NULL;
PFNGLFRAGMENTLIGHTIEXTPROC __glewFragmentLightiEXT = NULL;
PFNGLFRAGMENTLIGHTIVEXTPROC __glewFragmentLightivEXT = NULL;
PFNGLFRAGMENTMATERIALFEXTPROC __glewFragmentMaterialfEXT = NULL;
PFNGLFRAGMENTMATERIALFVEXTPROC __glewFragmentMaterialfvEXT = NULL;
PFNGLFRAGMENTMATERIALIEXTPROC __glewFragmentMaterialiEXT = NULL;
PFNGLFRAGMENTMATERIALIVEXTPROC __glewFragmentMaterialivEXT = NULL;
PFNGLGETFRAGMENTLIGHTFVEXTPROC __glewGetFragmentLightfvEXT = NULL;
PFNGLGETFRAGMENTLIGHTIVEXTPROC __glewGetFragmentLightivEXT = NULL;
PFNGLGETFRAGMENTMATERIALFVEXTPROC __glewGetFragmentMaterialfvEXT = NULL;
PFNGLGETFRAGMENTMATERIALIVEXTPROC __glewGetFragmentMaterialivEXT = NULL;
PFNGLLIGHTENVIEXTPROC __glewLightEnviEXT = NULL;

PFNGLBLITFRAMEBUFFEREXTPROC __glewBlitFramebufferEXT = NULL;

PFNGLRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC __glewRenderbufferStorageMultisampleEXT = NULL;

PFNGLBINDFRAMEBUFFEREXTPROC __glewBindFramebufferEXT = NULL;
PFNGLBINDRENDERBUFFEREXTPROC __glewBindRenderbufferEXT = NULL;
PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC __glewCheckFramebufferStatusEXT = NULL;
PFNGLDELETEFRAMEBUFFERSEXTPROC __glewDeleteFramebuffersEXT = NULL;
PFNGLDELETERENDERBUFFERSEXTPROC __glewDeleteRenderbuffersEXT = NULL;
PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC __glewFramebufferRenderbufferEXT = NULL;
PFNGLFRAMEBUFFERTEXTURE1DEXTPROC __glewFramebufferTexture1DEXT = NULL;
PFNGLFRAMEBUFFERTEXTURE2DEXTPROC __glewFramebufferTexture2DEXT = NULL;
PFNGLFRAMEBUFFERTEXTURE3DEXTPROC __glewFramebufferTexture3DEXT = NULL;
PFNGLGENFRAMEBUFFERSEXTPROC __glewGenFramebuffersEXT = NULL;
PFNGLGENRENDERBUFFERSEXTPROC __glewGenRenderbuffersEXT = NULL;
PFNGLGENERATEMIPMAPEXTPROC __glewGenerateMipmapEXT = NULL;
PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC __glewGetFramebufferAttachmentParameterivEXT = NULL;
PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC __glewGetRenderbufferParameterivEXT = NULL;
PFNGLISFRAMEBUFFEREXTPROC __glewIsFramebufferEXT = NULL;
PFNGLISRENDERBUFFEREXTPROC __glewIsRenderbufferEXT = NULL;
PFNGLRENDERBUFFERSTORAGEEXTPROC __glewRenderbufferStorageEXT = NULL;

PFNGLFRAMEBUFFERTEXTUREEXTPROC __glewFramebufferTextureEXT = NULL;
PFNGLFRAMEBUFFERTEXTUREFACEEXTPROC __glewFramebufferTextureFaceEXT = NULL;
PFNGLPROGRAMPARAMETERIEXTPROC __glewProgramParameteriEXT = NULL;

PFNGLPROGRAMENVPARAMETERS4FVEXTPROC __glewProgramEnvParameters4fvEXT = NULL;
PFNGLPROGRAMLOCALPARAMETERS4FVEXTPROC __glewProgramLocalParameters4fvEXT = NULL;

PFNGLBINDFRAGDATALOCATIONEXTPROC __glewBindFragDataLocationEXT = NULL;
PFNGLGETFRAGDATALOCATIONEXTPROC __glewGetFragDataLocationEXT = NULL;
PFNGLGETUNIFORMUIVEXTPROC __glewGetUniformuivEXT = NULL;
PFNGLGETVERTEXATTRIBIIVEXTPROC __glewGetVertexAttribIivEXT = NULL;
PFNGLGETVERTEXATTRIBIUIVEXTPROC __glewGetVertexAttribIuivEXT = NULL;
PFNGLUNIFORM1UIEXTPROC __glewUniform1uiEXT = NULL;
PFNGLUNIFORM1UIVEXTPROC __glewUniform1uivEXT = NULL;
PFNGLUNIFORM2UIEXTPROC __glewUniform2uiEXT = NULL;
PFNGLUNIFORM2UIVEXTPROC __glewUniform2uivEXT = NULL;
PFNGLUNIFORM3UIEXTPROC __glewUniform3uiEXT = NULL;
PFNGLUNIFORM3UIVEXTPROC __glewUniform3uivEXT = NULL;
PFNGLUNIFORM4UIEXTPROC __glewUniform4uiEXT = NULL;
PFNGLUNIFORM4UIVEXTPROC __glewUniform4uivEXT = NULL;
PFNGLVERTEXATTRIBI1IEXTPROC __glewVertexAttribI1iEXT = NULL;
PFNGLVERTEXATTRIBI1IVEXTPROC __glewVertexAttribI1ivEXT = NULL;
PFNGLVERTEXATTRIBI1UIEXTPROC __glewVertexAttribI1uiEXT = NULL;
PFNGLVERTEXATTRIBI1UIVEXTPROC __glewVertexAttribI1uivEXT = NULL;
PFNGLVERTEXATTRIBI2IEXTPROC __glewVertexAttribI2iEXT = NULL;
PFNGLVERTEXATTRIBI2IVEXTPROC __glewVertexAttribI2ivEXT = NULL;
PFNGLVERTEXATTRIBI2UIEXTPROC __glewVertexAttribI2uiEXT = NULL;
PFNGLVERTEXATTRIBI2UIVEXTPROC __glewVertexAttribI2uivEXT = NULL;
PFNGLVERTEXATTRIBI3IEXTPROC __glewVertexAttribI3iEXT = NULL;
PFNGLVERTEXATTRIBI3IVEXTPROC __glewVertexAttribI3ivEXT = NULL;
PFNGLVERTEXATTRIBI3UIEXTPROC __glewVertexAttribI3uiEXT = NULL;
PFNGLVERTEXATTRIBI3UIVEXTPROC __glewVertexAttribI3uivEXT = NULL;
PFNGLVERTEXATTRIBI4BVEXTPROC __glewVertexAttribI4bvEXT = NULL;
PFNGLVERTEXATTRIBI4IEXTPROC __glewVertexAttribI4iEXT = NULL;
PFNGLVERTEXATTRIBI4IVEXTPROC __glewVertexAttribI4ivEXT = NULL;
PFNGLVERTEXATTRIBI4SVEXTPROC __glewVertexAttribI4svEXT = NULL;
PFNGLVERTEXATTRIBI4UBVEXTPROC __glewVertexAttribI4ubvEXT = NULL;
PFNGLVERTEXATTRIBI4UIEXTPROC __glewVertexAttribI4uiEXT = NULL;
PFNGLVERTEXATTRIBI4UIVEXTPROC __glewVertexAttribI4uivEXT = NULL;
PFNGLVERTEXATTRIBI4USVEXTPROC __glewVertexAttribI4usvEXT = NULL;
PFNGLVERTEXATTRIBIPOINTEREXTPROC __glewVertexAttribIPointerEXT = NULL;

PFNGLGETHISTOGRAMEXTPROC __glewGetHistogramEXT = NULL;
PFNGLGETHISTOGRAMPARAMETERFVEXTPROC __glewGetHistogramParameterfvEXT = NULL;
PFNGLGETHISTOGRAMPARAMETERIVEXTPROC __glewGetHistogramParameterivEXT = NULL;
PFNGLGETMINMAXEXTPROC __glewGetMinmaxEXT = NULL;
PFNGLGETMINMAXPARAMETERFVEXTPROC __glewGetMinmaxParameterfvEXT = NULL;
PFNGLGETMINMAXPARAMETERIVEXTPROC __glewGetMinmaxParameterivEXT = NULL;
PFNGLHISTOGRAMEXTPROC __glewHistogramEXT = NULL;
PFNGLMINMAXEXTPROC __glewMinmaxEXT = NULL;
PFNGLRESETHISTOGRAMEXTPROC __glewResetHistogramEXT = NULL;
PFNGLRESETMINMAXEXTPROC __glewResetMinmaxEXT = NULL;

PFNGLINDEXFUNCEXTPROC __glewIndexFuncEXT = NULL;

PFNGLINDEXMATERIALEXTPROC __glewIndexMaterialEXT = NULL;

PFNGLAPPLYTEXTUREEXTPROC __glewApplyTextureEXT = NULL;
PFNGLTEXTURELIGHTEXTPROC __glewTextureLightEXT = NULL;
PFNGLTEXTUREMATERIALEXTPROC __glewTextureMaterialEXT = NULL;

PFNGLMULTIDRAWARRAYSEXTPROC __glewMultiDrawArraysEXT = NULL;
PFNGLMULTIDRAWELEMENTSEXTPROC __glewMultiDrawElementsEXT = NULL;

PFNGLSAMPLEMASKEXTPROC __glewSampleMaskEXT = NULL;
PFNGLSAMPLEPATTERNEXTPROC __glewSamplePatternEXT = NULL;

PFNGLCOLORTABLEEXTPROC __glewColorTableEXT = NULL;
PFNGLGETCOLORTABLEEXTPROC __glewGetColorTableEXT = NULL;
PFNGLGETCOLORTABLEPARAMETERFVEXTPROC __glewGetColorTableParameterfvEXT = NULL;
PFNGLGETCOLORTABLEPARAMETERIVEXTPROC __glewGetColorTableParameterivEXT = NULL;

PFNGLGETPIXELTRANSFORMPARAMETERFVEXTPROC __glewGetPixelTransformParameterfvEXT = NULL;
PFNGLGETPIXELTRANSFORMPARAMETERIVEXTPROC __glewGetPixelTransformParameterivEXT = NULL;
PFNGLPIXELTRANSFORMPARAMETERFEXTPROC __glewPixelTransformParameterfEXT = NULL;
PFNGLPIXELTRANSFORMPARAMETERFVEXTPROC __glewPixelTransformParameterfvEXT = NULL;
PFNGLPIXELTRANSFORMPARAMETERIEXTPROC __glewPixelTransformParameteriEXT = NULL;
PFNGLPIXELTRANSFORMPARAMETERIVEXTPROC __glewPixelTransformParameterivEXT = NULL;

PFNGLPOINTPARAMETERFEXTPROC __glewPointParameterfEXT = NULL;
PFNGLPOINTPARAMETERFVEXTPROC __glewPointParameterfvEXT = NULL;

PFNGLPOLYGONOFFSETEXTPROC __glewPolygonOffsetEXT = NULL;

PFNGLPOLYGONOFFSETCLAMPEXTPROC __glewPolygonOffsetClampEXT = NULL;

PFNGLPROVOKINGVERTEXEXTPROC __glewProvokingVertexEXT = NULL;

PFNGLCOVERAGEMODULATIONNVPROC __glewCoverageModulationNV = NULL;
PFNGLCOVERAGEMODULATIONTABLENVPROC __glewCoverageModulationTableNV = NULL;
PFNGLGETCOVERAGEMODULATIONTABLENVPROC __glewGetCoverageModulationTableNV = NULL;
PFNGLRASTERSAMPLESEXTPROC __glewRasterSamplesEXT = NULL;

PFNGLBEGINSCENEEXTPROC __glewBeginSceneEXT = NULL;
PFNGLENDSCENEEXTPROC __glewEndSceneEXT = NULL;

PFNGLSECONDARYCOLOR3BEXTPROC __glewSecondaryColor3bEXT = NULL;
PFNGLSECONDARYCOLOR3BVEXTPROC __glewSecondaryColor3bvEXT = NULL;
PFNGLSECONDARYCOLOR3DEXTPROC __glewSecondaryColor3dEXT = NULL;
PFNGLSECONDARYCOLOR3DVEXTPROC __glewSecondaryColor3dvEXT = NULL;
PFNGLSECONDARYCOLOR3FEXTPROC __glewSecondaryColor3fEXT = NULL;
PFNGLSECONDARYCOLOR3FVEXTPROC __glewSecondaryColor3fvEXT = NULL;
PFNGLSECONDARYCOLOR3IEXTPROC __glewSecondaryColor3iEXT = NULL;
PFNGLSECONDARYCOLOR3IVEXTPROC __glewSecondaryColor3ivEXT = NULL;
PFNGLSECONDARYCOLOR3SEXTPROC __glewSecondaryColor3sEXT = NULL;
PFNGLSECONDARYCOLOR3SVEXTPROC __glewSecondaryColor3svEXT = NULL;
PFNGLSECONDARYCOLOR3UBEXTPROC __glewSecondaryColor3ubEXT = NULL;
PFNGLSECONDARYCOLOR3UBVEXTPROC __glewSecondaryColor3ubvEXT = NULL;
PFNGLSECONDARYCOLOR3UIEXTPROC __glewSecondaryColor3uiEXT = NULL;
PFNGLSECONDARYCOLOR3UIVEXTPROC __glewSecondaryColor3uivEXT = NULL;
PFNGLSECONDARYCOLOR3USEXTPROC __glewSecondaryColor3usEXT = NULL;
PFNGLSECONDARYCOLOR3USVEXTPROC __glewSecondaryColor3usvEXT = NULL;
PFNGLSECONDARYCOLORPOINTEREXTPROC __glewSecondaryColorPointerEXT = NULL;

PFNGLACTIVEPROGRAMEXTPROC __glewActiveProgramEXT = NULL;
PFNGLCREATESHADERPROGRAMEXTPROC __glewCreateShaderProgramEXT = NULL;
PFNGLUSESHADERPROGRAMEXTPROC __glewUseShaderProgramEXT = NULL;

PFNGLBINDIMAGETEXTUREEXTPROC __glewBindImageTextureEXT = NULL;
PFNGLMEMORYBARRIEREXTPROC __glewMemoryBarrierEXT = NULL;

PFNGLACTIVESTENCILFACEEXTPROC __glewActiveStencilFaceEXT = NULL;

PFNGLTEXSUBIMAGE1DEXTPROC __glewTexSubImage1DEXT = NULL;
PFNGLTEXSUBIMAGE2DEXTPROC __glewTexSubImage2DEXT = NULL;
PFNGLTEXSUBIMAGE3DEXTPROC __glewTexSubImage3DEXT = NULL;

PFNGLTEXIMAGE3DEXTPROC __glewTexImage3DEXT = NULL;

PFNGLFRAMEBUFFERTEXTURELAYEREXTPROC __glewFramebufferTextureLayerEXT = NULL;

PFNGLTEXBUFFEREXTPROC __glewTexBufferEXT = NULL;

PFNGLCLEARCOLORIIEXTPROC __glewClearColorIiEXT = NULL;
PFNGLCLEARCOLORIUIEXTPROC __glewClearColorIuiEXT = NULL;
PFNGLGETTEXPARAMETERIIVEXTPROC __glewGetTexParameterIivEXT = NULL;
PFNGLGETTEXPARAMETERIUIVEXTPROC __glewGetTexParameterIuivEXT = NULL;
PFNGLTEXPARAMETERIIVEXTPROC __glewTexParameterIivEXT = NULL;
PFNGLTEXPARAMETERIUIVEXTPROC __glewTexParameterIuivEXT = NULL;

PFNGLARETEXTURESRESIDENTEXTPROC __glewAreTexturesResidentEXT = NULL;
PFNGLBINDTEXTUREEXTPROC __glewBindTextureEXT = NULL;
PFNGLDELETETEXTURESEXTPROC __glewDeleteTexturesEXT = NULL;
PFNGLGENTEXTURESEXTPROC __glewGenTexturesEXT = NULL;
PFNGLISTEXTUREEXTPROC __glewIsTextureEXT = NULL;
PFNGLPRIORITIZETEXTURESEXTPROC __glewPrioritizeTexturesEXT = NULL;

PFNGLTEXTURENORMALEXTPROC __glewTextureNormalEXT = NULL;

PFNGLGETQUERYOBJECTI64VEXTPROC __glewGetQueryObjecti64vEXT = NULL;
PFNGLGETQUERYOBJECTUI64VEXTPROC __glewGetQueryObjectui64vEXT = NULL;

PFNGLBEGINTRANSFORMFEEDBACKEXTPROC __glewBeginTransformFeedbackEXT = NULL;
PFNGLBINDBUFFERBASEEXTPROC __glewBindBufferBaseEXT = NULL;
PFNGLBINDBUFFEROFFSETEXTPROC __glewBindBufferOffsetEXT = NULL;
PFNGLBINDBUFFERRANGEEXTPROC __glewBindBufferRangeEXT = NULL;
PFNGLENDTRANSFORMFEEDBACKEXTPROC __glewEndTransformFeedbackEXT = NULL;
PFNGLGETTRANSFORMFEEDBACKVARYINGEXTPROC __glewGetTransformFeedbackVaryingEXT = NULL;
PFNGLTRANSFORMFEEDBACKVARYINGSEXTPROC __glewTransformFeedbackVaryingsEXT = NULL;

PFNGLARRAYELEMENTEXTPROC __glewArrayElementEXT = NULL;
PFNGLCOLORPOINTEREXTPROC __glewColorPointerEXT = NULL;
PFNGLDRAWARRAYSEXTPROC __glewDrawArraysEXT = NULL;
PFNGLEDGEFLAGPOINTEREXTPROC __glewEdgeFlagPointerEXT = NULL;
PFNGLINDEXPOINTEREXTPROC __glewIndexPointerEXT = NULL;
PFNGLNORMALPOINTEREXTPROC __glewNormalPointerEXT = NULL;
PFNGLTEXCOORDPOINTEREXTPROC __glewTexCoordPointerEXT = NULL;
PFNGLVERTEXPOINTEREXTPROC __glewVertexPointerEXT = NULL;

PFNGLGETVERTEXATTRIBLDVEXTPROC __glewGetVertexAttribLdvEXT = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBLOFFSETEXTPROC __glewVertexArrayVertexAttribLOffsetEXT = NULL;
PFNGLVERTEXATTRIBL1DEXTPROC __glewVertexAttribL1dEXT = NULL;
PFNGLVERTEXATTRIBL1DVEXTPROC __glewVertexAttribL1dvEXT = NULL;
PFNGLVERTEXATTRIBL2DEXTPROC __glewVertexAttribL2dEXT = NULL;
PFNGLVERTEXATTRIBL2DVEXTPROC __glewVertexAttribL2dvEXT = NULL;
PFNGLVERTEXATTRIBL3DEXTPROC __glewVertexAttribL3dEXT = NULL;
PFNGLVERTEXATTRIBL3DVEXTPROC __glewVertexAttribL3dvEXT = NULL;
PFNGLVERTEXATTRIBL4DEXTPROC __glewVertexAttribL4dEXT = NULL;
PFNGLVERTEXATTRIBL4DVEXTPROC __glewVertexAttribL4dvEXT = NULL;
PFNGLVERTEXATTRIBLPOINTEREXTPROC __glewVertexAttribLPointerEXT = NULL;

PFNGLBEGINVERTEXSHADEREXTPROC __glewBeginVertexShaderEXT = NULL;
PFNGLBINDLIGHTPARAMETEREXTPROC __glewBindLightParameterEXT = NULL;
PFNGLBINDMATERIALPARAMETEREXTPROC __glewBindMaterialParameterEXT = NULL;
PFNGLBINDPARAMETEREXTPROC __glewBindParameterEXT = NULL;
PFNGLBINDTEXGENPARAMETEREXTPROC __glewBindTexGenParameterEXT = NULL;
PFNGLBINDTEXTUREUNITPARAMETEREXTPROC __glewBindTextureUnitParameterEXT = NULL;
PFNGLBINDVERTEXSHADEREXTPROC __glewBindVertexShaderEXT = NULL;
PFNGLDELETEVERTEXSHADEREXTPROC __glewDeleteVertexShaderEXT = NULL;
PFNGLDISABLEVARIANTCLIENTSTATEEXTPROC __glewDisableVariantClientStateEXT = NULL;
PFNGLENABLEVARIANTCLIENTSTATEEXTPROC __glewEnableVariantClientStateEXT = NULL;
PFNGLENDVERTEXSHADEREXTPROC __glewEndVertexShaderEXT = NULL;
PFNGLEXTRACTCOMPONENTEXTPROC __glewExtractComponentEXT = NULL;
PFNGLGENSYMBOLSEXTPROC __glewGenSymbolsEXT = NULL;
PFNGLGENVERTEXSHADERSEXTPROC __glewGenVertexShadersEXT = NULL;
PFNGLGETINVARIANTBOOLEANVEXTPROC __glewGetInvariantBooleanvEXT = NULL;
PFNGLGETINVARIANTFLOATVEXTPROC __glewGetInvariantFloatvEXT = NULL;
PFNGLGETINVARIANTINTEGERVEXTPROC __glewGetInvariantIntegervEXT = NULL;
PFNGLGETLOCALCONSTANTBOOLEANVEXTPROC __glewGetLocalConstantBooleanvEXT = NULL;
PFNGLGETLOCALCONSTANTFLOATVEXTPROC __glewGetLocalConstantFloatvEXT = NULL;
PFNGLGETLOCALCONSTANTINTEGERVEXTPROC __glewGetLocalConstantIntegervEXT = NULL;
PFNGLGETVARIANTBOOLEANVEXTPROC __glewGetVariantBooleanvEXT = NULL;
PFNGLGETVARIANTFLOATVEXTPROC __glewGetVariantFloatvEXT = NULL;
PFNGLGETVARIANTINTEGERVEXTPROC __glewGetVariantIntegervEXT = NULL;
PFNGLGETVARIANTPOINTERVEXTPROC __glewGetVariantPointervEXT = NULL;
PFNGLINSERTCOMPONENTEXTPROC __glewInsertComponentEXT = NULL;
PFNGLISVARIANTENABLEDEXTPROC __glewIsVariantEnabledEXT = NULL;
PFNGLSETINVARIANTEXTPROC __glewSetInvariantEXT = NULL;
PFNGLSETLOCALCONSTANTEXTPROC __glewSetLocalConstantEXT = NULL;
PFNGLSHADEROP1EXTPROC __glewShaderOp1EXT = NULL;
PFNGLSHADEROP2EXTPROC __glewShaderOp2EXT = NULL;
PFNGLSHADEROP3EXTPROC __glewShaderOp3EXT = NULL;
PFNGLSWIZZLEEXTPROC __glewSwizzleEXT = NULL;
PFNGLVARIANTPOINTEREXTPROC __glewVariantPointerEXT = NULL;
PFNGLVARIANTBVEXTPROC __glewVariantbvEXT = NULL;
PFNGLVARIANTDVEXTPROC __glewVariantdvEXT = NULL;
PFNGLVARIANTFVEXTPROC __glewVariantfvEXT = NULL;
PFNGLVARIANTIVEXTPROC __glewVariantivEXT = NULL;
PFNGLVARIANTSVEXTPROC __glewVariantsvEXT = NULL;
PFNGLVARIANTUBVEXTPROC __glewVariantubvEXT = NULL;
PFNGLVARIANTUIVEXTPROC __glewVariantuivEXT = NULL;
PFNGLVARIANTUSVEXTPROC __glewVariantusvEXT = NULL;
PFNGLWRITEMASKEXTPROC __glewWriteMaskEXT = NULL;

PFNGLVERTEXWEIGHTPOINTEREXTPROC __glewVertexWeightPointerEXT = NULL;
PFNGLVERTEXWEIGHTFEXTPROC __glewVertexWeightfEXT = NULL;
PFNGLVERTEXWEIGHTFVEXTPROC __glewVertexWeightfvEXT = NULL;

PFNGLIMPORTSYNCEXTPROC __glewImportSyncEXT = NULL;

PFNGLFRAMETERMINATORGREMEDYPROC __glewFrameTerminatorGREMEDY = NULL;

PFNGLSTRINGMARKERGREMEDYPROC __glewStringMarkerGREMEDY = NULL;

PFNGLGETIMAGETRANSFORMPARAMETERFVHPPROC __glewGetImageTransformParameterfvHP = NULL;
PFNGLGETIMAGETRANSFORMPARAMETERIVHPPROC __glewGetImageTransformParameterivHP = NULL;
PFNGLIMAGETRANSFORMPARAMETERFHPPROC __glewImageTransformParameterfHP = NULL;
PFNGLIMAGETRANSFORMPARAMETERFVHPPROC __glewImageTransformParameterfvHP = NULL;
PFNGLIMAGETRANSFORMPARAMETERIHPPROC __glewImageTransformParameteriHP = NULL;
PFNGLIMAGETRANSFORMPARAMETERIVHPPROC __glewImageTransformParameterivHP = NULL;

PFNGLMULTIMODEDRAWARRAYSIBMPROC __glewMultiModeDrawArraysIBM = NULL;
PFNGLMULTIMODEDRAWELEMENTSIBMPROC __glewMultiModeDrawElementsIBM = NULL;

PFNGLCOLORPOINTERLISTIBMPROC __glewColorPointerListIBM = NULL;
PFNGLEDGEFLAGPOINTERLISTIBMPROC __glewEdgeFlagPointerListIBM = NULL;
PFNGLFOGCOORDPOINTERLISTIBMPROC __glewFogCoordPointerListIBM = NULL;
PFNGLINDEXPOINTERLISTIBMPROC __glewIndexPointerListIBM = NULL;
PFNGLNORMALPOINTERLISTIBMPROC __glewNormalPointerListIBM = NULL;
PFNGLSECONDARYCOLORPOINTERLISTIBMPROC __glewSecondaryColorPointerListIBM = NULL;
PFNGLTEXCOORDPOINTERLISTIBMPROC __glewTexCoordPointerListIBM = NULL;
PFNGLVERTEXPOINTERLISTIBMPROC __glewVertexPointerListIBM = NULL;

PFNGLMAPTEXTURE2DINTELPROC __glewMapTexture2DINTEL = NULL;
PFNGLSYNCTEXTUREINTELPROC __glewSyncTextureINTEL = NULL;
PFNGLUNMAPTEXTURE2DINTELPROC __glewUnmapTexture2DINTEL = NULL;

PFNGLCOLORPOINTERVINTELPROC __glewColorPointervINTEL = NULL;
PFNGLNORMALPOINTERVINTELPROC __glewNormalPointervINTEL = NULL;
PFNGLTEXCOORDPOINTERVINTELPROC __glewTexCoordPointervINTEL = NULL;
PFNGLVERTEXPOINTERVINTELPROC __glewVertexPointervINTEL = NULL;

PFNGLBEGINPERFQUERYINTELPROC __glewBeginPerfQueryINTEL = NULL;
PFNGLCREATEPERFQUERYINTELPROC __glewCreatePerfQueryINTEL = NULL;
PFNGLDELETEPERFQUERYINTELPROC __glewDeletePerfQueryINTEL = NULL;
PFNGLENDPERFQUERYINTELPROC __glewEndPerfQueryINTEL = NULL;
PFNGLGETFIRSTPERFQUERYIDINTELPROC __glewGetFirstPerfQueryIdINTEL = NULL;
PFNGLGETNEXTPERFQUERYIDINTELPROC __glewGetNextPerfQueryIdINTEL = NULL;
PFNGLGETPERFCOUNTERINFOINTELPROC __glewGetPerfCounterInfoINTEL = NULL;
PFNGLGETPERFQUERYDATAINTELPROC __glewGetPerfQueryDataINTEL = NULL;
PFNGLGETPERFQUERYIDBYNAMEINTELPROC __glewGetPerfQueryIdByNameINTEL = NULL;
PFNGLGETPERFQUERYINFOINTELPROC __glewGetPerfQueryInfoINTEL = NULL;

PFNGLTEXSCISSORFUNCINTELPROC __glewTexScissorFuncINTEL = NULL;
PFNGLTEXSCISSORINTELPROC __glewTexScissorINTEL = NULL;

PFNGLBLENDBARRIERKHRPROC __glewBlendBarrierKHR = NULL;

PFNGLDEBUGMESSAGECALLBACKPROC __glewDebugMessageCallback = NULL;
PFNGLDEBUGMESSAGECONTROLPROC __glewDebugMessageControl = NULL;
PFNGLDEBUGMESSAGEINSERTPROC __glewDebugMessageInsert = NULL;
PFNGLGETDEBUGMESSAGELOGPROC __glewGetDebugMessageLog = NULL;
PFNGLGETOBJECTLABELPROC __glewGetObjectLabel = NULL;
PFNGLGETOBJECTPTRLABELPROC __glewGetObjectPtrLabel = NULL;
PFNGLOBJECTLABELPROC __glewObjectLabel = NULL;
PFNGLOBJECTPTRLABELPROC __glewObjectPtrLabel = NULL;
PFNGLPOPDEBUGGROUPPROC __glewPopDebugGroup = NULL;
PFNGLPUSHDEBUGGROUPPROC __glewPushDebugGroup = NULL;

PFNGLGETNUNIFORMFVPROC __glewGetnUniformfv = NULL;
PFNGLGETNUNIFORMIVPROC __glewGetnUniformiv = NULL;
PFNGLGETNUNIFORMUIVPROC __glewGetnUniformuiv = NULL;
PFNGLREADNPIXELSPROC __glewReadnPixels = NULL;

PFNGLBUFFERREGIONENABLEDPROC __glewBufferRegionEnabled = NULL;
PFNGLDELETEBUFFERREGIONPROC __glewDeleteBufferRegion = NULL;
PFNGLDRAWBUFFERREGIONPROC __glewDrawBufferRegion = NULL;
PFNGLNEWBUFFERREGIONPROC __glewNewBufferRegion = NULL;
PFNGLREADBUFFERREGIONPROC __glewReadBufferRegion = NULL;

PFNGLRESIZEBUFFERSMESAPROC __glewResizeBuffersMESA = NULL;

PFNGLWINDOWPOS2DMESAPROC __glewWindowPos2dMESA = NULL;
PFNGLWINDOWPOS2DVMESAPROC __glewWindowPos2dvMESA = NULL;
PFNGLWINDOWPOS2FMESAPROC __glewWindowPos2fMESA = NULL;
PFNGLWINDOWPOS2FVMESAPROC __glewWindowPos2fvMESA = NULL;
PFNGLWINDOWPOS2IMESAPROC __glewWindowPos2iMESA = NULL;
PFNGLWINDOWPOS2IVMESAPROC __glewWindowPos2ivMESA = NULL;
PFNGLWINDOWPOS2SMESAPROC __glewWindowPos2sMESA = NULL;
PFNGLWINDOWPOS2SVMESAPROC __glewWindowPos2svMESA = NULL;
PFNGLWINDOWPOS3DMESAPROC __glewWindowPos3dMESA = NULL;
PFNGLWINDOWPOS3DVMESAPROC __glewWindowPos3dvMESA = NULL;
PFNGLWINDOWPOS3FMESAPROC __glewWindowPos3fMESA = NULL;
PFNGLWINDOWPOS3FVMESAPROC __glewWindowPos3fvMESA = NULL;
PFNGLWINDOWPOS3IMESAPROC __glewWindowPos3iMESA = NULL;
PFNGLWINDOWPOS3IVMESAPROC __glewWindowPos3ivMESA = NULL;
PFNGLWINDOWPOS3SMESAPROC __glewWindowPos3sMESA = NULL;
PFNGLWINDOWPOS3SVMESAPROC __glewWindowPos3svMESA = NULL;
PFNGLWINDOWPOS4DMESAPROC __glewWindowPos4dMESA = NULL;
PFNGLWINDOWPOS4DVMESAPROC __glewWindowPos4dvMESA = NULL;
PFNGLWINDOWPOS4FMESAPROC __glewWindowPos4fMESA = NULL;
PFNGLWINDOWPOS4FVMESAPROC __glewWindowPos4fvMESA = NULL;
PFNGLWINDOWPOS4IMESAPROC __glewWindowPos4iMESA = NULL;
PFNGLWINDOWPOS4IVMESAPROC __glewWindowPos4ivMESA = NULL;
PFNGLWINDOWPOS4SMESAPROC __glewWindowPos4sMESA = NULL;
PFNGLWINDOWPOS4SVMESAPROC __glewWindowPos4svMESA = NULL;

PFNGLBEGINCONDITIONALRENDERNVXPROC __glewBeginConditionalRenderNVX = NULL;
PFNGLENDCONDITIONALRENDERNVXPROC __glewEndConditionalRenderNVX = NULL;

PFNGLMULTIDRAWARRAYSINDIRECTBINDLESSNVPROC __glewMultiDrawArraysIndirectBindlessNV = NULL;
PFNGLMULTIDRAWELEMENTSINDIRECTBINDLESSNVPROC __glewMultiDrawElementsIndirectBindlessNV = NULL;

PFNGLMULTIDRAWARRAYSINDIRECTBINDLESSCOUNTNVPROC __glewMultiDrawArraysIndirectBindlessCountNV = NULL;
PFNGLMULTIDRAWELEMENTSINDIRECTBINDLESSCOUNTNVPROC __glewMultiDrawElementsIndirectBindlessCountNV = NULL;

PFNGLGETIMAGEHANDLENVPROC __glewGetImageHandleNV = NULL;
PFNGLGETTEXTUREHANDLENVPROC __glewGetTextureHandleNV = NULL;
PFNGLGETTEXTURESAMPLERHANDLENVPROC __glewGetTextureSamplerHandleNV = NULL;
PFNGLISIMAGEHANDLERESIDENTNVPROC __glewIsImageHandleResidentNV = NULL;
PFNGLISTEXTUREHANDLERESIDENTNVPROC __glewIsTextureHandleResidentNV = NULL;
PFNGLMAKEIMAGEHANDLENONRESIDENTNVPROC __glewMakeImageHandleNonResidentNV = NULL;
PFNGLMAKEIMAGEHANDLERESIDENTNVPROC __glewMakeImageHandleResidentNV = NULL;
PFNGLMAKETEXTUREHANDLENONRESIDENTNVPROC __glewMakeTextureHandleNonResidentNV = NULL;
PFNGLMAKETEXTUREHANDLERESIDENTNVPROC __glewMakeTextureHandleResidentNV = NULL;
PFNGLPROGRAMUNIFORMHANDLEUI64NVPROC __glewProgramUniformHandleui64NV = NULL;
PFNGLPROGRAMUNIFORMHANDLEUI64VNVPROC __glewProgramUniformHandleui64vNV = NULL;
PFNGLUNIFORMHANDLEUI64NVPROC __glewUniformHandleui64NV = NULL;
PFNGLUNIFORMHANDLEUI64VNVPROC __glewUniformHandleui64vNV = NULL;

PFNGLBLENDBARRIERNVPROC __glewBlendBarrierNV = NULL;
PFNGLBLENDPARAMETERINVPROC __glewBlendParameteriNV = NULL;

PFNGLBEGINCONDITIONALRENDERNVPROC __glewBeginConditionalRenderNV = NULL;
PFNGLENDCONDITIONALRENDERNVPROC __glewEndConditionalRenderNV = NULL;

PFNGLSUBPIXELPRECISIONBIASNVPROC __glewSubpixelPrecisionBiasNV = NULL;

PFNGLCONSERVATIVERASTERPARAMETERFNVPROC __glewConservativeRasterParameterfNV = NULL;

PFNGLCOPYIMAGESUBDATANVPROC __glewCopyImageSubDataNV = NULL;

PFNGLCLEARDEPTHDNVPROC __glewClearDepthdNV = NULL;
PFNGLDEPTHBOUNDSDNVPROC __glewDepthBoundsdNV = NULL;
PFNGLDEPTHRANGEDNVPROC __glewDepthRangedNV = NULL;

PFNGLDRAWTEXTURENVPROC __glewDrawTextureNV = NULL;

PFNGLEVALMAPSNVPROC __glewEvalMapsNV = NULL;
PFNGLGETMAPATTRIBPARAMETERFVNVPROC __glewGetMapAttribParameterfvNV = NULL;
PFNGLGETMAPATTRIBPARAMETERIVNVPROC __glewGetMapAttribParameterivNV = NULL;
PFNGLGETMAPCONTROLPOINTSNVPROC __glewGetMapControlPointsNV = NULL;
PFNGLGETMAPPARAMETERFVNVPROC __glewGetMapParameterfvNV = NULL;
PFNGLGETMAPPARAMETERIVNVPROC __glewGetMapParameterivNV = NULL;
PFNGLMAPCONTROLPOINTSNVPROC __glewMapControlPointsNV = NULL;
PFNGLMAPPARAMETERFVNVPROC __glewMapParameterfvNV = NULL;
PFNGLMAPPARAMETERIVNVPROC __glewMapParameterivNV = NULL;

PFNGLGETMULTISAMPLEFVNVPROC __glewGetMultisamplefvNV = NULL;
PFNGLSAMPLEMASKINDEXEDNVPROC __glewSampleMaskIndexedNV = NULL;
PFNGLTEXRENDERBUFFERNVPROC __glewTexRenderbufferNV = NULL;

PFNGLDELETEFENCESNVPROC __glewDeleteFencesNV = NULL;
PFNGLFINISHFENCENVPROC __glewFinishFenceNV = NULL;
PFNGLGENFENCESNVPROC __glewGenFencesNV = NULL;
PFNGLGETFENCEIVNVPROC __glewGetFenceivNV = NULL;
PFNGLISFENCENVPROC __glewIsFenceNV = NULL;
PFNGLSETFENCENVPROC __glewSetFenceNV = NULL;
PFNGLTESTFENCENVPROC __glewTestFenceNV = NULL;

PFNGLFRAGMENTCOVERAGECOLORNVPROC __glewFragmentCoverageColorNV = NULL;

PFNGLGETPROGRAMNAMEDPARAMETERDVNVPROC __glewGetProgramNamedParameterdvNV = NULL;
PFNGLGETPROGRAMNAMEDPARAMETERFVNVPROC __glewGetProgramNamedParameterfvNV = NULL;
PFNGLPROGRAMNAMEDPARAMETER4DNVPROC __glewProgramNamedParameter4dNV = NULL;
PFNGLPROGRAMNAMEDPARAMETER4DVNVPROC __glewProgramNamedParameter4dvNV = NULL;
PFNGLPROGRAMNAMEDPARAMETER4FNVPROC __glewProgramNamedParameter4fNV = NULL;
PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC __glewProgramNamedParameter4fvNV = NULL;

PFNGLRENDERBUFFERSTORAGEMULTISAMPLECOVERAGENVPROC __glewRenderbufferStorageMultisampleCoverageNV = NULL;

PFNGLPROGRAMVERTEXLIMITNVPROC __glewProgramVertexLimitNV = NULL;

PFNGLPROGRAMENVPARAMETERI4INVPROC __glewProgramEnvParameterI4iNV = NULL;
PFNGLPROGRAMENVPARAMETERI4IVNVPROC __glewProgramEnvParameterI4ivNV = NULL;
PFNGLPROGRAMENVPARAMETERI4UINVPROC __glewProgramEnvParameterI4uiNV = NULL;
PFNGLPROGRAMENVPARAMETERI4UIVNVPROC __glewProgramEnvParameterI4uivNV = NULL;
PFNGLPROGRAMENVPARAMETERSI4IVNVPROC __glewProgramEnvParametersI4ivNV = NULL;
PFNGLPROGRAMENVPARAMETERSI4UIVNVPROC __glewProgramEnvParametersI4uivNV = NULL;
PFNGLPROGRAMLOCALPARAMETERI4INVPROC __glewProgramLocalParameterI4iNV = NULL;
PFNGLPROGRAMLOCALPARAMETERI4IVNVPROC __glewProgramLocalParameterI4ivNV = NULL;
PFNGLPROGRAMLOCALPARAMETERI4UINVPROC __glewProgramLocalParameterI4uiNV = NULL;
PFNGLPROGRAMLOCALPARAMETERI4UIVNVPROC __glewProgramLocalParameterI4uivNV = NULL;
PFNGLPROGRAMLOCALPARAMETERSI4IVNVPROC __glewProgramLocalParametersI4ivNV = NULL;
PFNGLPROGRAMLOCALPARAMETERSI4UIVNVPROC __glewProgramLocalParametersI4uivNV = NULL;

PFNGLGETUNIFORMI64VNVPROC __glewGetUniformi64vNV = NULL;
PFNGLGETUNIFORMUI64VNVPROC __glewGetUniformui64vNV = NULL;
PFNGLPROGRAMUNIFORM1I64NVPROC __glewProgramUniform1i64NV = NULL;
PFNGLPROGRAMUNIFORM1I64VNVPROC __glewProgramUniform1i64vNV = NULL;
PFNGLPROGRAMUNIFORM1UI64NVPROC __glewProgramUniform1ui64NV = NULL;
PFNGLPROGRAMUNIFORM1UI64VNVPROC __glewProgramUniform1ui64vNV = NULL;
PFNGLPROGRAMUNIFORM2I64NVPROC __glewProgramUniform2i64NV = NULL;
PFNGLPROGRAMUNIFORM2I64VNVPROC __glewProgramUniform2i64vNV = NULL;
PFNGLPROGRAMUNIFORM2UI64NVPROC __glewProgramUniform2ui64NV = NULL;
PFNGLPROGRAMUNIFORM2UI64VNVPROC __glewProgramUniform2ui64vNV = NULL;
PFNGLPROGRAMUNIFORM3I64NVPROC __glewProgramUniform3i64NV = NULL;
PFNGLPROGRAMUNIFORM3I64VNVPROC __glewProgramUniform3i64vNV = NULL;
PFNGLPROGRAMUNIFORM3UI64NVPROC __glewProgramUniform3ui64NV = NULL;
PFNGLPROGRAMUNIFORM3UI64VNVPROC __glewProgramUniform3ui64vNV = NULL;
PFNGLPROGRAMUNIFORM4I64NVPROC __glewProgramUniform4i64NV = NULL;
PFNGLPROGRAMUNIFORM4I64VNVPROC __glewProgramUniform4i64vNV = NULL;
PFNGLPROGRAMUNIFORM4UI64NVPROC __glewProgramUniform4ui64NV = NULL;
PFNGLPROGRAMUNIFORM4UI64VNVPROC __glewProgramUniform4ui64vNV = NULL;
PFNGLUNIFORM1I64NVPROC __glewUniform1i64NV = NULL;
PFNGLUNIFORM1I64VNVPROC __glewUniform1i64vNV = NULL;
PFNGLUNIFORM1UI64NVPROC __glewUniform1ui64NV = NULL;
PFNGLUNIFORM1UI64VNVPROC __glewUniform1ui64vNV = NULL;
PFNGLUNIFORM2I64NVPROC __glewUniform2i64NV = NULL;
PFNGLUNIFORM2I64VNVPROC __glewUniform2i64vNV = NULL;
PFNGLUNIFORM2UI64NVPROC __glewUniform2ui64NV = NULL;
PFNGLUNIFORM2UI64VNVPROC __glewUniform2ui64vNV = NULL;
PFNGLUNIFORM3I64NVPROC __glewUniform3i64NV = NULL;
PFNGLUNIFORM3I64VNVPROC __glewUniform3i64vNV = NULL;
PFNGLUNIFORM3UI64NVPROC __glewUniform3ui64NV = NULL;
PFNGLUNIFORM3UI64VNVPROC __glewUniform3ui64vNV = NULL;
PFNGLUNIFORM4I64NVPROC __glewUniform4i64NV = NULL;
PFNGLUNIFORM4I64VNVPROC __glewUniform4i64vNV = NULL;
PFNGLUNIFORM4UI64NVPROC __glewUniform4ui64NV = NULL;
PFNGLUNIFORM4UI64VNVPROC __glewUniform4ui64vNV = NULL;

PFNGLCOLOR3HNVPROC __glewColor3hNV = NULL;
PFNGLCOLOR3HVNVPROC __glewColor3hvNV = NULL;
PFNGLCOLOR4HNVPROC __glewColor4hNV = NULL;
PFNGLCOLOR4HVNVPROC __glewColor4hvNV = NULL;
PFNGLFOGCOORDHNVPROC __glewFogCoordhNV = NULL;
PFNGLFOGCOORDHVNVPROC __glewFogCoordhvNV = NULL;
PFNGLMULTITEXCOORD1HNVPROC __glewMultiTexCoord1hNV = NULL;
PFNGLMULTITEXCOORD1HVNVPROC __glewMultiTexCoord1hvNV = NULL;
PFNGLMULTITEXCOORD2HNVPROC __glewMultiTexCoord2hNV = NULL;
PFNGLMULTITEXCOORD2HVNVPROC __glewMultiTexCoord2hvNV = NULL;
PFNGLMULTITEXCOORD3HNVPROC __glewMultiTexCoord3hNV = NULL;
PFNGLMULTITEXCOORD3HVNVPROC __glewMultiTexCoord3hvNV = NULL;
PFNGLMULTITEXCOORD4HNVPROC __glewMultiTexCoord4hNV = NULL;
PFNGLMULTITEXCOORD4HVNVPROC __glewMultiTexCoord4hvNV = NULL;
PFNGLNORMAL3HNVPROC __glewNormal3hNV = NULL;
PFNGLNORMAL3HVNVPROC __glewNormal3hvNV = NULL;
PFNGLSECONDARYCOLOR3HNVPROC __glewSecondaryColor3hNV = NULL;
PFNGLSECONDARYCOLOR3HVNVPROC __glewSecondaryColor3hvNV = NULL;
PFNGLTEXCOORD1HNVPROC __glewTexCoord1hNV = NULL;
PFNGLTEXCOORD1HVNVPROC __glewTexCoord1hvNV = NULL;
PFNGLTEXCOORD2HNVPROC __glewTexCoord2hNV = NULL;
PFNGLTEXCOORD2HVNVPROC __glewTexCoord2hvNV = NULL;
PFNGLTEXCOORD3HNVPROC __glewTexCoord3hNV = NULL;
PFNGLTEXCOORD3HVNVPROC __glewTexCoord3hvNV = NULL;
PFNGLTEXCOORD4HNVPROC __glewTexCoord4hNV = NULL;
PFNGLTEXCOORD4HVNVPROC __glewTexCoord4hvNV = NULL;
PFNGLVERTEX2HNVPROC __glewVertex2hNV = NULL;
PFNGLVERTEX2HVNVPROC __glewVertex2hvNV = NULL;
PFNGLVERTEX3HNVPROC __glewVertex3hNV = NULL;
PFNGLVERTEX3HVNVPROC __glewVertex3hvNV = NULL;
PFNGLVERTEX4HNVPROC __glewVertex4hNV = NULL;
PFNGLVERTEX4HVNVPROC __glewVertex4hvNV = NULL;
PFNGLVERTEXATTRIB1HNVPROC __glewVertexAttrib1hNV = NULL;
PFNGLVERTEXATTRIB1HVNVPROC __glewVertexAttrib1hvNV = NULL;
PFNGLVERTEXATTRIB2HNVPROC __glewVertexAttrib2hNV = NULL;
PFNGLVERTEXATTRIB2HVNVPROC __glewVertexAttrib2hvNV = NULL;
PFNGLVERTEXATTRIB3HNVPROC __glewVertexAttrib3hNV = NULL;
PFNGLVERTEXATTRIB3HVNVPROC __glewVertexAttrib3hvNV = NULL;
PFNGLVERTEXATTRIB4HNVPROC __glewVertexAttrib4hNV = NULL;
PFNGLVERTEXATTRIB4HVNVPROC __glewVertexAttrib4hvNV = NULL;
PFNGLVERTEXATTRIBS1HVNVPROC __glewVertexAttribs1hvNV = NULL;
PFNGLVERTEXATTRIBS2HVNVPROC __glewVertexAttribs2hvNV = NULL;
PFNGLVERTEXATTRIBS3HVNVPROC __glewVertexAttribs3hvNV = NULL;
PFNGLVERTEXATTRIBS4HVNVPROC __glewVertexAttribs4hvNV = NULL;
PFNGLVERTEXWEIGHTHNVPROC __glewVertexWeighthNV = NULL;
PFNGLVERTEXWEIGHTHVNVPROC __glewVertexWeighthvNV = NULL;

PFNGLGETINTERNALFORMATSAMPLEIVNVPROC __glewGetInternalformatSampleivNV = NULL;

PFNGLBEGINOCCLUSIONQUERYNVPROC __glewBeginOcclusionQueryNV = NULL;
PFNGLDELETEOCCLUSIONQUERIESNVPROC __glewDeleteOcclusionQueriesNV = NULL;
PFNGLENDOCCLUSIONQUERYNVPROC __glewEndOcclusionQueryNV = NULL;
PFNGLGENOCCLUSIONQUERIESNVPROC __glewGenOcclusionQueriesNV = NULL;
PFNGLGETOCCLUSIONQUERYIVNVPROC __glewGetOcclusionQueryivNV = NULL;
PFNGLGETOCCLUSIONQUERYUIVNVPROC __glewGetOcclusionQueryuivNV = NULL;
PFNGLISOCCLUSIONQUERYNVPROC __glewIsOcclusionQueryNV = NULL;

PFNGLPROGRAMBUFFERPARAMETERSIIVNVPROC __glewProgramBufferParametersIivNV = NULL;
PFNGLPROGRAMBUFFERPARAMETERSIUIVNVPROC __glewProgramBufferParametersIuivNV = NULL;
PFNGLPROGRAMBUFFERPARAMETERSFVNVPROC __glewProgramBufferParametersfvNV = NULL;

PFNGLCOPYPATHNVPROC __glewCopyPathNV = NULL;
PFNGLCOVERFILLPATHINSTANCEDNVPROC __glewCoverFillPathInstancedNV = NULL;
PFNGLCOVERFILLPATHNVPROC __glewCoverFillPathNV = NULL;
PFNGLCOVERSTROKEPATHINSTANCEDNVPROC __glewCoverStrokePathInstancedNV = NULL;
PFNGLCOVERSTROKEPATHNVPROC __glewCoverStrokePathNV = NULL;
PFNGLDELETEPATHSNVPROC __glewDeletePathsNV = NULL;
PFNGLGENPATHSNVPROC __glewGenPathsNV = NULL;
PFNGLGETPATHCOLORGENFVNVPROC __glewGetPathColorGenfvNV = NULL;
PFNGLGETPATHCOLORGENIVNVPROC __glewGetPathColorGenivNV = NULL;
PFNGLGETPATHCOMMANDSNVPROC __glewGetPathCommandsNV = NULL;
PFNGLGETPATHCOORDSNVPROC __glewGetPathCoordsNV = NULL;
PFNGLGETPATHDASHARRAYNVPROC __glewGetPathDashArrayNV = NULL;
PFNGLGETPATHLENGTHNVPROC __glewGetPathLengthNV = NULL;
PFNGLGETPATHMETRICRANGENVPROC __glewGetPathMetricRangeNV = NULL;
PFNGLGETPATHMETRICSNVPROC __glewGetPathMetricsNV = NULL;
PFNGLGETPATHPARAMETERFVNVPROC __glewGetPathParameterfvNV = NULL;
PFNGLGETPATHPARAMETERIVNVPROC __glewGetPathParameterivNV = NULL;
PFNGLGETPATHSPACINGNVPROC __glewGetPathSpacingNV = NULL;
PFNGLGETPATHTEXGENFVNVPROC __glewGetPathTexGenfvNV = NULL;
PFNGLGETPATHTEXGENIVNVPROC __glewGetPathTexGenivNV = NULL;
PFNGLGETPROGRAMRESOURCEFVNVPROC __glewGetProgramResourcefvNV = NULL;
PFNGLINTERPOLATEPATHSNVPROC __glewInterpolatePathsNV = NULL;
PFNGLISPATHNVPROC __glewIsPathNV = NULL;
PFNGLISPOINTINFILLPATHNVPROC __glewIsPointInFillPathNV = NULL;
PFNGLISPOINTINSTROKEPATHNVPROC __glewIsPointInStrokePathNV = NULL;
PFNGLMATRIXLOAD3X2FNVPROC __glewMatrixLoad3x2fNV = NULL;
PFNGLMATRIXLOAD3X3FNVPROC __glewMatrixLoad3x3fNV = NULL;
PFNGLMATRIXLOADTRANSPOSE3X3FNVPROC __glewMatrixLoadTranspose3x3fNV = NULL;
PFNGLMATRIXMULT3X2FNVPROC __glewMatrixMult3x2fNV = NULL;
PFNGLMATRIXMULT3X3FNVPROC __glewMatrixMult3x3fNV = NULL;
PFNGLMATRIXMULTTRANSPOSE3X3FNVPROC __glewMatrixMultTranspose3x3fNV = NULL;
PFNGLPATHCOLORGENNVPROC __glewPathColorGenNV = NULL;
PFNGLPATHCOMMANDSNVPROC __glewPathCommandsNV = NULL;
PFNGLPATHCOORDSNVPROC __glewPathCoordsNV = NULL;
PFNGLPATHCOVERDEPTHFUNCNVPROC __glewPathCoverDepthFuncNV = NULL;
PFNGLPATHDASHARRAYNVPROC __glewPathDashArrayNV = NULL;
PFNGLPATHFOGGENNVPROC __glewPathFogGenNV = NULL;
PFNGLPATHGLYPHINDEXARRAYNVPROC __glewPathGlyphIndexArrayNV = NULL;
PFNGLPATHGLYPHINDEXRANGENVPROC __glewPathGlyphIndexRangeNV = NULL;
PFNGLPATHGLYPHRANGENVPROC __glewPathGlyphRangeNV = NULL;
PFNGLPATHGLYPHSNVPROC __glewPathGlyphsNV = NULL;
PFNGLPATHMEMORYGLYPHINDEXARRAYNVPROC __glewPathMemoryGlyphIndexArrayNV = NULL;
PFNGLPATHPARAMETERFNVPROC __glewPathParameterfNV = NULL;
PFNGLPATHPARAMETERFVNVPROC __glewPathParameterfvNV = NULL;
PFNGLPATHPARAMETERINVPROC __glewPathParameteriNV = NULL;
PFNGLPATHPARAMETERIVNVPROC __glewPathParameterivNV = NULL;
PFNGLPATHSTENCILDEPTHOFFSETNVPROC __glewPathStencilDepthOffsetNV = NULL;
PFNGLPATHSTENCILFUNCNVPROC __glewPathStencilFuncNV = NULL;
PFNGLPATHSTRINGNVPROC __glewPathStringNV = NULL;
PFNGLPATHSUBCOMMANDSNVPROC __glewPathSubCommandsNV = NULL;
PFNGLPATHSUBCOORDSNVPROC __glewPathSubCoordsNV = NULL;
PFNGLPATHTEXGENNVPROC __glewPathTexGenNV = NULL;
PFNGLPOINTALONGPATHNVPROC __glewPointAlongPathNV = NULL;
PFNGLPROGRAMPATHFRAGMENTINPUTGENNVPROC __glewProgramPathFragmentInputGenNV = NULL;
PFNGLSTENCILFILLPATHINSTANCEDNVPROC __glewStencilFillPathInstancedNV = NULL;
PFNGLSTENCILFILLPATHNVPROC __glewStencilFillPathNV = NULL;
PFNGLSTENCILSTROKEPATHINSTANCEDNVPROC __glewStencilStrokePathInstancedNV = NULL;
PFNGLSTENCILSTROKEPATHNVPROC __glewStencilStrokePathNV = NULL;
PFNGLSTENCILTHENCOVERFILLPATHINSTANCEDNVPROC __glewStencilThenCoverFillPathInstancedNV = NULL;
PFNGLSTENCILTHENCOVERFILLPATHNVPROC __glewStencilThenCoverFillPathNV = NULL;
PFNGLSTENCILTHENCOVERSTROKEPATHINSTANCEDNVPROC __glewStencilThenCoverStrokePathInstancedNV = NULL;
PFNGLSTENCILTHENCOVERSTROKEPATHNVPROC __glewStencilThenCoverStrokePathNV = NULL;
PFNGLTRANSFORMPATHNVPROC __glewTransformPathNV = NULL;
PFNGLWEIGHTPATHSNVPROC __glewWeightPathsNV = NULL;

PFNGLFLUSHPIXELDATARANGENVPROC __glewFlushPixelDataRangeNV = NULL;
PFNGLPIXELDATARANGENVPROC __glewPixelDataRangeNV = NULL;

PFNGLPOINTPARAMETERINVPROC __glewPointParameteriNV = NULL;
PFNGLPOINTPARAMETERIVNVPROC __glewPointParameterivNV = NULL;

PFNGLGETVIDEOI64VNVPROC __glewGetVideoi64vNV = NULL;
PFNGLGETVIDEOIVNVPROC __glewGetVideoivNV = NULL;
PFNGLGETVIDEOUI64VNVPROC __glewGetVideoui64vNV = NULL;
PFNGLGETVIDEOUIVNVPROC __glewGetVideouivNV = NULL;
PFNGLPRESENTFRAMEDUALFILLNVPROC __glewPresentFrameDualFillNV = NULL;
PFNGLPRESENTFRAMEKEYEDNVPROC __glewPresentFrameKeyedNV = NULL;

PFNGLPRIMITIVERESTARTINDEXNVPROC __glewPrimitiveRestartIndexNV = NULL;
PFNGLPRIMITIVERESTARTNVPROC __glewPrimitiveRestartNV = NULL;

PFNGLCOMBINERINPUTNVPROC __glewCombinerInputNV = NULL;
PFNGLCOMBINEROUTPUTNVPROC __glewCombinerOutputNV = NULL;
PFNGLCOMBINERPARAMETERFNVPROC __glewCombinerParameterfNV = NULL;
PFNGLCOMBINERPARAMETERFVNVPROC __glewCombinerParameterfvNV = NULL;
PFNGLCOMBINERPARAMETERINVPROC __glewCombinerParameteriNV = NULL;
PFNGLCOMBINERPARAMETERIVNVPROC __glewCombinerParameterivNV = NULL;
PFNGLFINALCOMBINERINPUTNVPROC __glewFinalCombinerInputNV = NULL;
PFNGLGETCOMBINERINPUTPARAMETERFVNVPROC __glewGetCombinerInputParameterfvNV = NULL;
PFNGLGETCOMBINERINPUTPARAMETERIVNVPROC __glewGetCombinerInputParameterivNV = NULL;
PFNGLGETCOMBINEROUTPUTPARAMETERFVNVPROC __glewGetCombinerOutputParameterfvNV = NULL;
PFNGLGETCOMBINEROUTPUTPARAMETERIVNVPROC __glewGetCombinerOutputParameterivNV = NULL;
PFNGLGETFINALCOMBINERINPUTPARAMETERFVNVPROC __glewGetFinalCombinerInputParameterfvNV = NULL;
PFNGLGETFINALCOMBINERINPUTPARAMETERIVNVPROC __glewGetFinalCombinerInputParameterivNV = NULL;

PFNGLCOMBINERSTAGEPARAMETERFVNVPROC __glewCombinerStageParameterfvNV = NULL;
PFNGLGETCOMBINERSTAGEPARAMETERFVNVPROC __glewGetCombinerStageParameterfvNV = NULL;

PFNGLFRAMEBUFFERSAMPLELOCATIONSFVNVPROC __glewFramebufferSampleLocationsfvNV = NULL;
PFNGLNAMEDFRAMEBUFFERSAMPLELOCATIONSFVNVPROC __glewNamedFramebufferSampleLocationsfvNV = NULL;

PFNGLGETBUFFERPARAMETERUI64VNVPROC __glewGetBufferParameterui64vNV = NULL;
PFNGLGETINTEGERUI64VNVPROC __glewGetIntegerui64vNV = NULL;
PFNGLGETNAMEDBUFFERPARAMETERUI64VNVPROC __glewGetNamedBufferParameterui64vNV = NULL;
PFNGLISBUFFERRESIDENTNVPROC __glewIsBufferResidentNV = NULL;
PFNGLISNAMEDBUFFERRESIDENTNVPROC __glewIsNamedBufferResidentNV = NULL;
PFNGLMAKEBUFFERNONRESIDENTNVPROC __glewMakeBufferNonResidentNV = NULL;
PFNGLMAKEBUFFERRESIDENTNVPROC __glewMakeBufferResidentNV = NULL;
PFNGLMAKENAMEDBUFFERNONRESIDENTNVPROC __glewMakeNamedBufferNonResidentNV = NULL;
PFNGLMAKENAMEDBUFFERRESIDENTNVPROC __glewMakeNamedBufferResidentNV = NULL;
PFNGLPROGRAMUNIFORMUI64NVPROC __glewProgramUniformui64NV = NULL;
PFNGLPROGRAMUNIFORMUI64VNVPROC __glewProgramUniformui64vNV = NULL;
PFNGLUNIFORMUI64NVPROC __glewUniformui64NV = NULL;
PFNGLUNIFORMUI64VNVPROC __glewUniformui64vNV = NULL;

PFNGLTEXTUREBARRIERNVPROC __glewTextureBarrierNV = NULL;

PFNGLTEXIMAGE2DMULTISAMPLECOVERAGENVPROC __glewTexImage2DMultisampleCoverageNV = NULL;
PFNGLTEXIMAGE3DMULTISAMPLECOVERAGENVPROC __glewTexImage3DMultisampleCoverageNV = NULL;
PFNGLTEXTUREIMAGE2DMULTISAMPLECOVERAGENVPROC __glewTextureImage2DMultisampleCoverageNV = NULL;
PFNGLTEXTUREIMAGE2DMULTISAMPLENVPROC __glewTextureImage2DMultisampleNV = NULL;
PFNGLTEXTUREIMAGE3DMULTISAMPLECOVERAGENVPROC __glewTextureImage3DMultisampleCoverageNV = NULL;
PFNGLTEXTUREIMAGE3DMULTISAMPLENVPROC __glewTextureImage3DMultisampleNV = NULL;

PFNGLACTIVEVARYINGNVPROC __glewActiveVaryingNV = NULL;
PFNGLBEGINTRANSFORMFEEDBACKNVPROC __glewBeginTransformFeedbackNV = NULL;
PFNGLBINDBUFFERBASENVPROC __glewBindBufferBaseNV = NULL;
PFNGLBINDBUFFEROFFSETNVPROC __glewBindBufferOffsetNV = NULL;
PFNGLBINDBUFFERRANGENVPROC __glewBindBufferRangeNV = NULL;
PFNGLENDTRANSFORMFEEDBACKNVPROC __glewEndTransformFeedbackNV = NULL;
PFNGLGETACTIVEVARYINGNVPROC __glewGetActiveVaryingNV = NULL;
PFNGLGETTRANSFORMFEEDBACKVARYINGNVPROC __glewGetTransformFeedbackVaryingNV = NULL;
PFNGLGETVARYINGLOCATIONNVPROC __glewGetVaryingLocationNV = NULL;
PFNGLTRANSFORMFEEDBACKATTRIBSNVPROC __glewTransformFeedbackAttribsNV = NULL;
PFNGLTRANSFORMFEEDBACKVARYINGSNVPROC __glewTransformFeedbackVaryingsNV = NULL;

PFNGLBINDTRANSFORMFEEDBACKNVPROC __glewBindTransformFeedbackNV = NULL;
PFNGLDELETETRANSFORMFEEDBACKSNVPROC __glewDeleteTransformFeedbacksNV = NULL;
PFNGLDRAWTRANSFORMFEEDBACKNVPROC __glewDrawTransformFeedbackNV = NULL;
PFNGLGENTRANSFORMFEEDBACKSNVPROC __glewGenTransformFeedbacksNV = NULL;
PFNGLISTRANSFORMFEEDBACKNVPROC __glewIsTransformFeedbackNV = NULL;
PFNGLPAUSETRANSFORMFEEDBACKNVPROC __glewPauseTransformFeedbackNV = NULL;
PFNGLRESUMETRANSFORMFEEDBACKNVPROC __glewResumeTransformFeedbackNV = NULL;

PFNGLVDPAUFININVPROC __glewVDPAUFiniNV = NULL;
PFNGLVDPAUGETSURFACEIVNVPROC __glewVDPAUGetSurfaceivNV = NULL;
PFNGLVDPAUINITNVPROC __glewVDPAUInitNV = NULL;
PFNGLVDPAUISSURFACENVPROC __glewVDPAUIsSurfaceNV = NULL;
PFNGLVDPAUMAPSURFACESNVPROC __glewVDPAUMapSurfacesNV = NULL;
PFNGLVDPAUREGISTEROUTPUTSURFACENVPROC __glewVDPAURegisterOutputSurfaceNV = NULL;
PFNGLVDPAUREGISTERVIDEOSURFACENVPROC __glewVDPAURegisterVideoSurfaceNV = NULL;
PFNGLVDPAUSURFACEACCESSNVPROC __glewVDPAUSurfaceAccessNV = NULL;
PFNGLVDPAUUNMAPSURFACESNVPROC __glewVDPAUUnmapSurfacesNV = NULL;
PFNGLVDPAUUNREGISTERSURFACENVPROC __glewVDPAUUnregisterSurfaceNV = NULL;

PFNGLFLUSHVERTEXARRAYRANGENVPROC __glewFlushVertexArrayRangeNV = NULL;
PFNGLVERTEXARRAYRANGENVPROC __glewVertexArrayRangeNV = NULL;

PFNGLGETVERTEXATTRIBLI64VNVPROC __glewGetVertexAttribLi64vNV = NULL;
PFNGLGETVERTEXATTRIBLUI64VNVPROC __glewGetVertexAttribLui64vNV = NULL;
PFNGLVERTEXATTRIBL1I64NVPROC __glewVertexAttribL1i64NV = NULL;
PFNGLVERTEXATTRIBL1I64VNVPROC __glewVertexAttribL1i64vNV = NULL;
PFNGLVERTEXATTRIBL1UI64NVPROC __glewVertexAttribL1ui64NV = NULL;
PFNGLVERTEXATTRIBL1UI64VNVPROC __glewVertexAttribL1ui64vNV = NULL;
PFNGLVERTEXATTRIBL2I64NVPROC __glewVertexAttribL2i64NV = NULL;
PFNGLVERTEXATTRIBL2I64VNVPROC __glewVertexAttribL2i64vNV = NULL;
PFNGLVERTEXATTRIBL2UI64NVPROC __glewVertexAttribL2ui64NV = NULL;
PFNGLVERTEXATTRIBL2UI64VNVPROC __glewVertexAttribL2ui64vNV = NULL;
PFNGLVERTEXATTRIBL3I64NVPROC __glewVertexAttribL3i64NV = NULL;
PFNGLVERTEXATTRIBL3I64VNVPROC __glewVertexAttribL3i64vNV = NULL;
PFNGLVERTEXATTRIBL3UI64NVPROC __glewVertexAttribL3ui64NV = NULL;
PFNGLVERTEXATTRIBL3UI64VNVPROC __glewVertexAttribL3ui64vNV = NULL;
PFNGLVERTEXATTRIBL4I64NVPROC __glewVertexAttribL4i64NV = NULL;
PFNGLVERTEXATTRIBL4I64VNVPROC __glewVertexAttribL4i64vNV = NULL;
PFNGLVERTEXATTRIBL4UI64NVPROC __glewVertexAttribL4ui64NV = NULL;
PFNGLVERTEXATTRIBL4UI64VNVPROC __glewVertexAttribL4ui64vNV = NULL;
PFNGLVERTEXATTRIBLFORMATNVPROC __glewVertexAttribLFormatNV = NULL;

PFNGLBUFFERADDRESSRANGENVPROC __glewBufferAddressRangeNV = NULL;
PFNGLCOLORFORMATNVPROC __glewColorFormatNV = NULL;
PFNGLEDGEFLAGFORMATNVPROC __glewEdgeFlagFormatNV = NULL;
PFNGLFOGCOORDFORMATNVPROC __glewFogCoordFormatNV = NULL;
PFNGLGETINTEGERUI64I_VNVPROC __glewGetIntegerui64i_vNV = NULL;
PFNGLINDEXFORMATNVPROC __glewIndexFormatNV = NULL;
PFNGLNORMALFORMATNVPROC __glewNormalFormatNV = NULL;
PFNGLSECONDARYCOLORFORMATNVPROC __glewSecondaryColorFormatNV = NULL;
PFNGLTEXCOORDFORMATNVPROC __glewTexCoordFormatNV = NULL;
PFNGLVERTEXATTRIBFORMATNVPROC __glewVertexAttribFormatNV = NULL;
PFNGLVERTEXATTRIBIFORMATNVPROC __glewVertexAttribIFormatNV = NULL;
PFNGLVERTEXFORMATNVPROC __glewVertexFormatNV = NULL;

PFNGLAREPROGRAMSRESIDENTNVPROC __glewAreProgramsResidentNV = NULL;
PFNGLBINDPROGRAMNVPROC __glewBindProgramNV = NULL;
PFNGLDELETEPROGRAMSNVPROC __glewDeleteProgramsNV = NULL;
PFNGLEXECUTEPROGRAMNVPROC __glewExecuteProgramNV = NULL;
PFNGLGENPROGRAMSNVPROC __glewGenProgramsNV = NULL;
PFNGLGETPROGRAMPARAMETERDVNVPROC __glewGetProgramParameterdvNV = NULL;
PFNGLGETPROGRAMPARAMETERFVNVPROC __glewGetProgramParameterfvNV = NULL;
PFNGLGETPROGRAMSTRINGNVPROC __glewGetProgramStringNV = NULL;
PFNGLGETPROGRAMIVNVPROC __glewGetProgramivNV = NULL;
PFNGLGETTRACKMATRIXIVNVPROC __glewGetTrackMatrixivNV = NULL;
PFNGLGETVERTEXATTRIBPOINTERVNVPROC __glewGetVertexAttribPointervNV = NULL;
PFNGLGETVERTEXATTRIBDVNVPROC __glewGetVertexAttribdvNV = NULL;
PFNGLGETVERTEXATTRIBFVNVPROC __glewGetVertexAttribfvNV = NULL;
PFNGLGETVERTEXATTRIBIVNVPROC __glewGetVertexAttribivNV = NULL;
PFNGLISPROGRAMNVPROC __glewIsProgramNV = NULL;
PFNGLLOADPROGRAMNVPROC __glewLoadProgramNV = NULL;
PFNGLPROGRAMPARAMETER4DNVPROC __glewProgramParameter4dNV = NULL;
PFNGLPROGRAMPARAMETER4DVNVPROC __glewProgramParameter4dvNV = NULL;
PFNGLPROGRAMPARAMETER4FNVPROC __glewProgramParameter4fNV = NULL;
PFNGLPROGRAMPARAMETER4FVNVPROC __glewProgramParameter4fvNV = NULL;
PFNGLPROGRAMPARAMETERS4DVNVPROC __glewProgramParameters4dvNV = NULL;
PFNGLPROGRAMPARAMETERS4FVNVPROC __glewProgramParameters4fvNV = NULL;
PFNGLREQUESTRESIDENTPROGRAMSNVPROC __glewRequestResidentProgramsNV = NULL;
PFNGLTRACKMATRIXNVPROC __glewTrackMatrixNV = NULL;
PFNGLVERTEXATTRIB1DNVPROC __glewVertexAttrib1dNV = NULL;
PFNGLVERTEXATTRIB1DVNVPROC __glewVertexAttrib1dvNV = NULL;
PFNGLVERTEXATTRIB1FNVPROC __glewVertexAttrib1fNV = NULL;
PFNGLVERTEXATTRIB1FVNVPROC __glewVertexAttrib1fvNV = NULL;
PFNGLVERTEXATTRIB1SNVPROC __glewVertexAttrib1sNV = NULL;
PFNGLVERTEXATTRIB1SVNVPROC __glewVertexAttrib1svNV = NULL;
PFNGLVERTEXATTRIB2DNVPROC __glewVertexAttrib2dNV = NULL;
PFNGLVERTEXATTRIB2DVNVPROC __glewVertexAttrib2dvNV = NULL;
PFNGLVERTEXATTRIB2FNVPROC __glewVertexAttrib2fNV = NULL;
PFNGLVERTEXATTRIB2FVNVPROC __glewVertexAttrib2fvNV = NULL;
PFNGLVERTEXATTRIB2SNVPROC __glewVertexAttrib2sNV = NULL;
PFNGLVERTEXATTRIB2SVNVPROC __glewVertexAttrib2svNV = NULL;
PFNGLVERTEXATTRIB3DNVPROC __glewVertexAttrib3dNV = NULL;
PFNGLVERTEXATTRIB3DVNVPROC __glewVertexAttrib3dvNV = NULL;
PFNGLVERTEXATTRIB3FNVPROC __glewVertexAttrib3fNV = NULL;
PFNGLVERTEXATTRIB3FVNVPROC __glewVertexAttrib3fvNV = NULL;
PFNGLVERTEXATTRIB3SNVPROC __glewVertexAttrib3sNV = NULL;
PFNGLVERTEXATTRIB3SVNVPROC __glewVertexAttrib3svNV = NULL;
PFNGLVERTEXATTRIB4DNVPROC __glewVertexAttrib4dNV = NULL;
PFNGLVERTEXATTRIB4DVNVPROC __glewVertexAttrib4dvNV = NULL;
PFNGLVERTEXATTRIB4FNVPROC __glewVertexAttrib4fNV = NULL;
PFNGLVERTEXATTRIB4FVNVPROC __glewVertexAttrib4fvNV = NULL;
PFNGLVERTEXATTRIB4SNVPROC __glewVertexAttrib4sNV = NULL;
PFNGLVERTEXATTRIB4SVNVPROC __glewVertexAttrib4svNV = NULL;
PFNGLVERTEXATTRIB4UBNVPROC __glewVertexAttrib4ubNV = NULL;
PFNGLVERTEXATTRIB4UBVNVPROC __glewVertexAttrib4ubvNV = NULL;
PFNGLVERTEXATTRIBPOINTERNVPROC __glewVertexAttribPointerNV = NULL;
PFNGLVERTEXATTRIBS1DVNVPROC __glewVertexAttribs1dvNV = NULL;
PFNGLVERTEXATTRIBS1FVNVPROC __glewVertexAttribs1fvNV = NULL;
PFNGLVERTEXATTRIBS1SVNVPROC __glewVertexAttribs1svNV = NULL;
PFNGLVERTEXATTRIBS2DVNVPROC __glewVertexAttribs2dvNV = NULL;
PFNGLVERTEXATTRIBS2FVNVPROC __glewVertexAttribs2fvNV = NULL;
PFNGLVERTEXATTRIBS2SVNVPROC __glewVertexAttribs2svNV = NULL;
PFNGLVERTEXATTRIBS3DVNVPROC __glewVertexAttribs3dvNV = NULL;
PFNGLVERTEXATTRIBS3FVNVPROC __glewVertexAttribs3fvNV = NULL;
PFNGLVERTEXATTRIBS3SVNVPROC __glewVertexAttribs3svNV = NULL;
PFNGLVERTEXATTRIBS4DVNVPROC __glewVertexAttribs4dvNV = NULL;
PFNGLVERTEXATTRIBS4FVNVPROC __glewVertexAttribs4fvNV = NULL;
PFNGLVERTEXATTRIBS4SVNVPROC __glewVertexAttribs4svNV = NULL;
PFNGLVERTEXATTRIBS4UBVNVPROC __glewVertexAttribs4ubvNV = NULL;

PFNGLBEGINVIDEOCAPTURENVPROC __glewBeginVideoCaptureNV = NULL;
PFNGLBINDVIDEOCAPTURESTREAMBUFFERNVPROC __glewBindVideoCaptureStreamBufferNV = NULL;
PFNGLBINDVIDEOCAPTURESTREAMTEXTURENVPROC __glewBindVideoCaptureStreamTextureNV = NULL;
PFNGLENDVIDEOCAPTURENVPROC __glewEndVideoCaptureNV = NULL;
PFNGLGETVIDEOCAPTURESTREAMDVNVPROC __glewGetVideoCaptureStreamdvNV = NULL;
PFNGLGETVIDEOCAPTURESTREAMFVNVPROC __glewGetVideoCaptureStreamfvNV = NULL;
PFNGLGETVIDEOCAPTURESTREAMIVNVPROC __glewGetVideoCaptureStreamivNV = NULL;
PFNGLGETVIDEOCAPTUREIVNVPROC __glewGetVideoCaptureivNV = NULL;
PFNGLVIDEOCAPTURENVPROC __glewVideoCaptureNV = NULL;
PFNGLVIDEOCAPTURESTREAMPARAMETERDVNVPROC __glewVideoCaptureStreamParameterdvNV = NULL;
PFNGLVIDEOCAPTURESTREAMPARAMETERFVNVPROC __glewVideoCaptureStreamParameterfvNV = NULL;
PFNGLVIDEOCAPTURESTREAMPARAMETERIVNVPROC __glewVideoCaptureStreamParameterivNV = NULL;

PFNGLCLEARDEPTHFOESPROC __glewClearDepthfOES = NULL;
PFNGLCLIPPLANEFOESPROC __glewClipPlanefOES = NULL;
PFNGLDEPTHRANGEFOESPROC __glewDepthRangefOES = NULL;
PFNGLFRUSTUMFOESPROC __glewFrustumfOES = NULL;
PFNGLGETCLIPPLANEFOESPROC __glewGetClipPlanefOES = NULL;
PFNGLORTHOFOESPROC __glewOrthofOES = NULL;

PFNGLFRAMEBUFFERTEXTUREMULTIVIEWOVRPROC __glewFramebufferTextureMultiviewOVR = NULL;

PFNGLALPHAFUNCXPROC __glewAlphaFuncx = NULL;
PFNGLCLEARCOLORXPROC __glewClearColorx = NULL;
PFNGLCLEARDEPTHXPROC __glewClearDepthx = NULL;
PFNGLCOLOR4XPROC __glewColor4x = NULL;
PFNGLDEPTHRANGEXPROC __glewDepthRangex = NULL;
PFNGLFOGXPROC __glewFogx = NULL;
PFNGLFOGXVPROC __glewFogxv = NULL;
PFNGLFRUSTUMFPROC __glewFrustumf = NULL;
PFNGLFRUSTUMXPROC __glewFrustumx = NULL;
PFNGLLIGHTMODELXPROC __glewLightModelx = NULL;
PFNGLLIGHTMODELXVPROC __glewLightModelxv = NULL;
PFNGLLIGHTXPROC __glewLightx = NULL;
PFNGLLIGHTXVPROC __glewLightxv = NULL;
PFNGLLINEWIDTHXPROC __glewLineWidthx = NULL;
PFNGLLOADMATRIXXPROC __glewLoadMatrixx = NULL;
PFNGLMATERIALXPROC __glewMaterialx = NULL;
PFNGLMATERIALXVPROC __glewMaterialxv = NULL;
PFNGLMULTMATRIXXPROC __glewMultMatrixx = NULL;
PFNGLMULTITEXCOORD4XPROC __glewMultiTexCoord4x = NULL;
PFNGLNORMAL3XPROC __glewNormal3x = NULL;
PFNGLORTHOFPROC __glewOrthof = NULL;
PFNGLORTHOXPROC __glewOrthox = NULL;
PFNGLPOINTSIZEXPROC __glewPointSizex = NULL;
PFNGLPOLYGONOFFSETXPROC __glewPolygonOffsetx = NULL;
PFNGLROTATEXPROC __glewRotatex = NULL;
PFNGLSAMPLECOVERAGEXPROC __glewSampleCoveragex = NULL;
PFNGLSCALEXPROC __glewScalex = NULL;
PFNGLTEXENVXPROC __glewTexEnvx = NULL;
PFNGLTEXENVXVPROC __glewTexEnvxv = NULL;
PFNGLTEXPARAMETERXPROC __glewTexParameterx = NULL;
PFNGLTRANSLATEXPROC __glewTranslatex = NULL;

PFNGLCLIPPLANEFPROC __glewClipPlanef = NULL;
PFNGLCLIPPLANEXPROC __glewClipPlanex = NULL;
PFNGLGETCLIPPLANEFPROC __glewGetClipPlanef = NULL;
PFNGLGETCLIPPLANEXPROC __glewGetClipPlanex = NULL;
PFNGLGETFIXEDVPROC __glewGetFixedv = NULL;
PFNGLGETLIGHTXVPROC __glewGetLightxv = NULL;
PFNGLGETMATERIALXVPROC __glewGetMaterialxv = NULL;
PFNGLGETTEXENVXVPROC __glewGetTexEnvxv = NULL;
PFNGLGETTEXPARAMETERXVPROC __glewGetTexParameterxv = NULL;
PFNGLPOINTPARAMETERXPROC __glewPointParameterx = NULL;
PFNGLPOINTPARAMETERXVPROC __glewPointParameterxv = NULL;
PFNGLPOINTSIZEPOINTEROESPROC __glewPointSizePointerOES = NULL;
PFNGLTEXPARAMETERXVPROC __glewTexParameterxv = NULL;

PFNGLERRORSTRINGREGALPROC __glewErrorStringREGAL = NULL;

PFNGLGETEXTENSIONREGALPROC __glewGetExtensionREGAL = NULL;
PFNGLISSUPPORTEDREGALPROC __glewIsSupportedREGAL = NULL;

PFNGLLOGMESSAGECALLBACKREGALPROC __glewLogMessageCallbackREGAL = NULL;

PFNGLGETPROCADDRESSREGALPROC __glewGetProcAddressREGAL = NULL;

PFNGLDETAILTEXFUNCSGISPROC __glewDetailTexFuncSGIS = NULL;
PFNGLGETDETAILTEXFUNCSGISPROC __glewGetDetailTexFuncSGIS = NULL;

PFNGLFOGFUNCSGISPROC __glewFogFuncSGIS = NULL;
PFNGLGETFOGFUNCSGISPROC __glewGetFogFuncSGIS = NULL;

PFNGLSAMPLEMASKSGISPROC __glewSampleMaskSGIS = NULL;
PFNGLSAMPLEPATTERNSGISPROC __glewSamplePatternSGIS = NULL;

PFNGLGETSHARPENTEXFUNCSGISPROC __glewGetSharpenTexFuncSGIS = NULL;
PFNGLSHARPENTEXFUNCSGISPROC __glewSharpenTexFuncSGIS = NULL;

PFNGLTEXIMAGE4DSGISPROC __glewTexImage4DSGIS = NULL;
PFNGLTEXSUBIMAGE4DSGISPROC __glewTexSubImage4DSGIS = NULL;

PFNGLGETTEXFILTERFUNCSGISPROC __glewGetTexFilterFuncSGIS = NULL;
PFNGLTEXFILTERFUNCSGISPROC __glewTexFilterFuncSGIS = NULL;

PFNGLASYNCMARKERSGIXPROC __glewAsyncMarkerSGIX = NULL;
PFNGLDELETEASYNCMARKERSSGIXPROC __glewDeleteAsyncMarkersSGIX = NULL;
PFNGLFINISHASYNCSGIXPROC __glewFinishAsyncSGIX = NULL;
PFNGLGENASYNCMARKERSSGIXPROC __glewGenAsyncMarkersSGIX = NULL;
PFNGLISASYNCMARKERSGIXPROC __glewIsAsyncMarkerSGIX = NULL;
PFNGLPOLLASYNCSGIXPROC __glewPollAsyncSGIX = NULL;

PFNGLFLUSHRASTERSGIXPROC __glewFlushRasterSGIX = NULL;

PFNGLTEXTUREFOGSGIXPROC __glewTextureFogSGIX = NULL;

PFNGLFRAGMENTCOLORMATERIALSGIXPROC __glewFragmentColorMaterialSGIX = NULL;
PFNGLFRAGMENTLIGHTMODELFSGIXPROC __glewFragmentLightModelfSGIX = NULL;
PFNGLFRAGMENTLIGHTMODELFVSGIXPROC __glewFragmentLightModelfvSGIX = NULL;
PFNGLFRAGMENTLIGHTMODELISGIXPROC __glewFragmentLightModeliSGIX = NULL;
PFNGLFRAGMENTLIGHTMODELIVSGIXPROC __glewFragmentLightModelivSGIX = NULL;
PFNGLFRAGMENTLIGHTFSGIXPROC __glewFragmentLightfSGIX = NULL;
PFNGLFRAGMENTLIGHTFVSGIXPROC __glewFragmentLightfvSGIX = NULL;
PFNGLFRAGMENTLIGHTISGIXPROC __glewFragmentLightiSGIX = NULL;
PFNGLFRAGMENTLIGHTIVSGIXPROC __glewFragmentLightivSGIX = NULL;
PFNGLFRAGMENTMATERIALFSGIXPROC __glewFragmentMaterialfSGIX = NULL;
PFNGLFRAGMENTMATERIALFVSGIXPROC __glewFragmentMaterialfvSGIX = NULL;
PFNGLFRAGMENTMATERIALISGIXPROC __glewFragmentMaterialiSGIX = NULL;
PFNGLFRAGMENTMATERIALIVSGIXPROC __glewFragmentMaterialivSGIX = NULL;
PFNGLGETFRAGMENTLIGHTFVSGIXPROC __glewGetFragmentLightfvSGIX = NULL;
PFNGLGETFRAGMENTLIGHTIVSGIXPROC __glewGetFragmentLightivSGIX = NULL;
PFNGLGETFRAGMENTMATERIALFVSGIXPROC __glewGetFragmentMaterialfvSGIX = NULL;
PFNGLGETFRAGMENTMATERIALIVSGIXPROC __glewGetFragmentMaterialivSGIX = NULL;

PFNGLFRAMEZOOMSGIXPROC __glewFrameZoomSGIX = NULL;

PFNGLPIXELTEXGENSGIXPROC __glewPixelTexGenSGIX = NULL;

PFNGLREFERENCEPLANESGIXPROC __glewReferencePlaneSGIX = NULL;

PFNGLSPRITEPARAMETERFSGIXPROC __glewSpriteParameterfSGIX = NULL;
PFNGLSPRITEPARAMETERFVSGIXPROC __glewSpriteParameterfvSGIX = NULL;
PFNGLSPRITEPARAMETERISGIXPROC __glewSpriteParameteriSGIX = NULL;
PFNGLSPRITEPARAMETERIVSGIXPROC __glewSpriteParameterivSGIX = NULL;

PFNGLTAGSAMPLEBUFFERSGIXPROC __glewTagSampleBufferSGIX = NULL;

PFNGLCOLORTABLEPARAMETERFVSGIPROC __glewColorTableParameterfvSGI = NULL;
PFNGLCOLORTABLEPARAMETERIVSGIPROC __glewColorTableParameterivSGI = NULL;
PFNGLCOLORTABLESGIPROC __glewColorTableSGI = NULL;
PFNGLCOPYCOLORTABLESGIPROC __glewCopyColorTableSGI = NULL;
PFNGLGETCOLORTABLEPARAMETERFVSGIPROC __glewGetColorTableParameterfvSGI = NULL;
PFNGLGETCOLORTABLEPARAMETERIVSGIPROC __glewGetColorTableParameterivSGI = NULL;
PFNGLGETCOLORTABLESGIPROC __glewGetColorTableSGI = NULL;

PFNGLFINISHTEXTURESUNXPROC __glewFinishTextureSUNX = NULL;

PFNGLGLOBALALPHAFACTORBSUNPROC __glewGlobalAlphaFactorbSUN = NULL;
PFNGLGLOBALALPHAFACTORDSUNPROC __glewGlobalAlphaFactordSUN = NULL;
PFNGLGLOBALALPHAFACTORFSUNPROC __glewGlobalAlphaFactorfSUN = NULL;
PFNGLGLOBALALPHAFACTORISUNPROC __glewGlobalAlphaFactoriSUN = NULL;
PFNGLGLOBALALPHAFACTORSSUNPROC __glewGlobalAlphaFactorsSUN = NULL;
PFNGLGLOBALALPHAFACTORUBSUNPROC __glewGlobalAlphaFactorubSUN = NULL;
PFNGLGLOBALALPHAFACTORUISUNPROC __glewGlobalAlphaFactoruiSUN = NULL;
PFNGLGLOBALALPHAFACTORUSSUNPROC __glewGlobalAlphaFactorusSUN = NULL;

PFNGLREADVIDEOPIXELSSUNPROC __glewReadVideoPixelsSUN = NULL;

PFNGLREPLACEMENTCODEPOINTERSUNPROC __glewReplacementCodePointerSUN = NULL;
PFNGLREPLACEMENTCODEUBSUNPROC __glewReplacementCodeubSUN = NULL;
PFNGLREPLACEMENTCODEUBVSUNPROC __glewReplacementCodeubvSUN = NULL;
PFNGLREPLACEMENTCODEUISUNPROC __glewReplacementCodeuiSUN = NULL;
PFNGLREPLACEMENTCODEUIVSUNPROC __glewReplacementCodeuivSUN = NULL;
PFNGLREPLACEMENTCODEUSSUNPROC __glewReplacementCodeusSUN = NULL;
PFNGLREPLACEMENTCODEUSVSUNPROC __glewReplacementCodeusvSUN = NULL;

PFNGLCOLOR3FVERTEX3FSUNPROC __glewColor3fVertex3fSUN = NULL;
PFNGLCOLOR3FVERTEX3FVSUNPROC __glewColor3fVertex3fvSUN = NULL;
PFNGLCOLOR4FNORMAL3FVERTEX3FSUNPROC __glewColor4fNormal3fVertex3fSUN = NULL;
PFNGLCOLOR4FNORMAL3FVERTEX3FVSUNPROC __glewColor4fNormal3fVertex3fvSUN = NULL;
PFNGLCOLOR4UBVERTEX2FSUNPROC __glewColor4ubVertex2fSUN = NULL;
PFNGLCOLOR4UBVERTEX2FVSUNPROC __glewColor4ubVertex2fvSUN = NULL;
PFNGLCOLOR4UBVERTEX3FSUNPROC __glewColor4ubVertex3fSUN = NULL;
PFNGLCOLOR4UBVERTEX3FVSUNPROC __glewColor4ubVertex3fvSUN = NULL;
PFNGLNORMAL3FVERTEX3FSUNPROC __glewNormal3fVertex3fSUN = NULL;
PFNGLNORMAL3FVERTEX3FVSUNPROC __glewNormal3fVertex3fvSUN = NULL;
PFNGLREPLACEMENTCODEUICOLOR3FVERTEX3FSUNPROC __glewReplacementCodeuiColor3fVertex3fSUN = NULL;
PFNGLREPLACEMENTCODEUICOLOR3FVERTEX3FVSUNPROC __glewReplacementCodeuiColor3fVertex3fvSUN = NULL;
PFNGLREPLACEMENTCODEUICOLOR4FNORMAL3FVERTEX3FSUNPROC __glewReplacementCodeuiColor4fNormal3fVertex3fSUN = NULL;
PFNGLREPLACEMENTCODEUICOLOR4FNORMAL3FVERTEX3FVSUNPROC __glewReplacementCodeuiColor4fNormal3fVertex3fvSUN = NULL;
PFNGLREPLACEMENTCODEUICOLOR4UBVERTEX3FSUNPROC __glewReplacementCodeuiColor4ubVertex3fSUN = NULL;
PFNGLREPLACEMENTCODEUICOLOR4UBVERTEX3FVSUNPROC __glewReplacementCodeuiColor4ubVertex3fvSUN = NULL;
PFNGLREPLACEMENTCODEUINORMAL3FVERTEX3FSUNPROC __glewReplacementCodeuiNormal3fVertex3fSUN = NULL;
PFNGLREPLACEMENTCODEUINORMAL3FVERTEX3FVSUNPROC __glewReplacementCodeuiNormal3fVertex3fvSUN = NULL;
PFNGLREPLACEMENTCODEUITEXCOORD2FCOLOR4FNORMAL3FVERTEX3FSUNPROC __glewReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fSUN = NULL;
PFNGLREPLACEMENTCODEUITEXCOORD2FCOLOR4FNORMAL3FVERTEX3FVSUNPROC __glewReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fvSUN = NULL;
PFNGLREPLACEMENTCODEUITEXCOORD2FNORMAL3FVERTEX3FSUNPROC __glewReplacementCodeuiTexCoord2fNormal3fVertex3fSUN = NULL;
PFNGLREPLACEMENTCODEUITEXCOORD2FNORMAL3FVERTEX3FVSUNPROC __glewReplacementCodeuiTexCoord2fNormal3fVertex3fvSUN = NULL;
PFNGLREPLACEMENTCODEUITEXCOORD2FVERTEX3FSUNPROC __glewReplacementCodeuiTexCoord2fVertex3fSUN = NULL;
PFNGLREPLACEMENTCODEUITEXCOORD2FVERTEX3FVSUNPROC __glewReplacementCodeuiTexCoord2fVertex3fvSUN = NULL;
PFNGLREPLACEMENTCODEUIVERTEX3FSUNPROC __glewReplacementCodeuiVertex3fSUN = NULL;
PFNGLREPLACEMENTCODEUIVERTEX3FVSUNPROC __glewReplacementCodeuiVertex3fvSUN = NULL;
PFNGLTEXCOORD2FCOLOR3FVERTEX3FSUNPROC __glewTexCoord2fColor3fVertex3fSUN = NULL;
PFNGLTEXCOORD2FCOLOR3FVERTEX3FVSUNPROC __glewTexCoord2fColor3fVertex3fvSUN = NULL;
PFNGLTEXCOORD2FCOLOR4FNORMAL3FVERTEX3FSUNPROC __glewTexCoord2fColor4fNormal3fVertex3fSUN = NULL;
PFNGLTEXCOORD2FCOLOR4FNORMAL3FVERTEX3FVSUNPROC __glewTexCoord2fColor4fNormal3fVertex3fvSUN = NULL;
PFNGLTEXCOORD2FCOLOR4UBVERTEX3FSUNPROC __glewTexCoord2fColor4ubVertex3fSUN = NULL;
PFNGLTEXCOORD2FCOLOR4UBVERTEX3FVSUNPROC __glewTexCoord2fColor4ubVertex3fvSUN = NULL;
PFNGLTEXCOORD2FNORMAL3FVERTEX3FSUNPROC __glewTexCoord2fNormal3fVertex3fSUN = NULL;
PFNGLTEXCOORD2FNORMAL3FVERTEX3FVSUNPROC __glewTexCoord2fNormal3fVertex3fvSUN = NULL;
PFNGLTEXCOORD2FVERTEX3FSUNPROC __glewTexCoord2fVertex3fSUN = NULL;
PFNGLTEXCOORD2FVERTEX3FVSUNPROC __glewTexCoord2fVertex3fvSUN = NULL;
PFNGLTEXCOORD4FCOLOR4FNORMAL3FVERTEX4FSUNPROC __glewTexCoord4fColor4fNormal3fVertex4fSUN = NULL;
PFNGLTEXCOORD4FCOLOR4FNORMAL3FVERTEX4FVSUNPROC __glewTexCoord4fColor4fNormal3fVertex4fvSUN = NULL;
PFNGLTEXCOORD4FVERTEX4FSUNPROC __glewTexCoord4fVertex4fSUN = NULL;
PFNGLTEXCOORD4FVERTEX4FVSUNPROC __glewTexCoord4fVertex4fvSUN = NULL;

PFNGLADDSWAPHINTRECTWINPROC __glewAddSwapHintRectWIN = NULL;

#endif /* !WIN32 || !GLEW_MX */

#if !defined(GLEW_MX)

GLboolean __GLEW_VERSION_1_1 = GL_FALSE;
GLboolean __GLEW_VERSION_1_2 = GL_FALSE;
GLboolean __GLEW_VERSION_1_2_1 = GL_FALSE;
GLboolean __GLEW_VERSION_1_3 = GL_FALSE;
GLboolean __GLEW_VERSION_1_4 = GL_FALSE;
GLboolean __GLEW_VERSION_1_5 = GL_FALSE;
GLboolean __GLEW_VERSION_2_0 = GL_FALSE;
GLboolean __GLEW_VERSION_2_1 = GL_FALSE;
GLboolean __GLEW_VERSION_3_0 = GL_FALSE;
GLboolean __GLEW_VERSION_3_1 = GL_FALSE;
GLboolean __GLEW_VERSION_3_2 = GL_FALSE;
GLboolean __GLEW_VERSION_3_3 = GL_FALSE;
GLboolean __GLEW_VERSION_4_0 = GL_FALSE;
GLboolean __GLEW_VERSION_4_1 = GL_FALSE;
GLboolean __GLEW_VERSION_4_2 = GL_FALSE;
GLboolean __GLEW_VERSION_4_3 = GL_FALSE;
GLboolean __GLEW_VERSION_4_4 = GL_FALSE;
GLboolean __GLEW_VERSION_4_5 = GL_FALSE;
GLboolean __GLEW_3DFX_multisample = GL_FALSE;
GLboolean __GLEW_3DFX_tbuffer = GL_FALSE;
GLboolean __GLEW_3DFX_texture_compression_FXT1 = GL_FALSE;
GLboolean __GLEW_AMD_blend_minmax_factor = GL_FALSE;
GLboolean __GLEW_AMD_conservative_depth = GL_FALSE;
GLboolean __GLEW_AMD_debug_output = GL_FALSE;
GLboolean __GLEW_AMD_depth_clamp_separate = GL_FALSE;
GLboolean __GLEW_AMD_draw_buffers_blend = GL_FALSE;
GLboolean __GLEW_AMD_gcn_shader = GL_FALSE;
GLboolean __GLEW_AMD_gpu_shader_int64 = GL_FALSE;
GLboolean __GLEW_AMD_interleaved_elements = GL_FALSE;
GLboolean __GLEW_AMD_multi_draw_indirect = GL_FALSE;
GLboolean __GLEW_AMD_name_gen_delete = GL_FALSE;
GLboolean __GLEW_AMD_occlusion_query_event = GL_FALSE;
GLboolean __GLEW_AMD_performance_monitor = GL_FALSE;
GLboolean __GLEW_AMD_pinned_memory = GL_FALSE;
GLboolean __GLEW_AMD_query_buffer_object = GL_FALSE;
GLboolean __GLEW_AMD_sample_positions = GL_FALSE;
GLboolean __GLEW_AMD_seamless_cubemap_per_texture = GL_FALSE;
GLboolean __GLEW_AMD_shader_atomic_counter_ops = GL_FALSE;
GLboolean __GLEW_AMD_shader_stencil_export = GL_FALSE;
GLboolean __GLEW_AMD_shader_stencil_value_export = GL_FALSE;
GLboolean __GLEW_AMD_shader_trinary_minmax = GL_FALSE;
GLboolean __GLEW_AMD_sparse_texture = GL_FALSE;
GLboolean __GLEW_AMD_stencil_operation_extended = GL_FALSE;
GLboolean __GLEW_AMD_texture_texture4 = GL_FALSE;
GLboolean __GLEW_AMD_transform_feedback3_lines_triangles = GL_FALSE;
GLboolean __GLEW_AMD_transform_feedback4 = GL_FALSE;
GLboolean __GLEW_AMD_vertex_shader_layer = GL_FALSE;
GLboolean __GLEW_AMD_vertex_shader_tessellator = GL_FALSE;
GLboolean __GLEW_AMD_vertex_shader_viewport_index = GL_FALSE;
GLboolean __GLEW_ANGLE_depth_texture = GL_FALSE;
GLboolean __GLEW_ANGLE_framebuffer_blit = GL_FALSE;
GLboolean __GLEW_ANGLE_framebuffer_multisample = GL_FALSE;
GLboolean __GLEW_ANGLE_instanced_arrays = GL_FALSE;
GLboolean __GLEW_ANGLE_pack_reverse_row_order = GL_FALSE;
GLboolean __GLEW_ANGLE_program_binary = GL_FALSE;
GLboolean __GLEW_ANGLE_texture_compression_dxt1 = GL_FALSE;
GLboolean __GLEW_ANGLE_texture_compression_dxt3 = GL_FALSE;
GLboolean __GLEW_ANGLE_texture_compression_dxt5 = GL_FALSE;
GLboolean __GLEW_ANGLE_texture_usage = GL_FALSE;
GLboolean __GLEW_ANGLE_timer_query = GL_FALSE;
GLboolean __GLEW_ANGLE_translated_shader_source = GL_FALSE;
GLboolean __GLEW_APPLE_aux_depth_stencil = GL_FALSE;
GLboolean __GLEW_APPLE_client_storage = GL_FALSE;
GLboolean __GLEW_APPLE_element_array = GL_FALSE;
GLboolean __GLEW_APPLE_fence = GL_FALSE;
GLboolean __GLEW_APPLE_float_pixels = GL_FALSE;
GLboolean __GLEW_APPLE_flush_buffer_range = GL_FALSE;
GLboolean __GLEW_APPLE_object_purgeable = GL_FALSE;
GLboolean __GLEW_APPLE_pixel_buffer = GL_FALSE;
GLboolean __GLEW_APPLE_rgb_422 = GL_FALSE;
GLboolean __GLEW_APPLE_row_bytes = GL_FALSE;
GLboolean __GLEW_APPLE_specular_vector = GL_FALSE;
GLboolean __GLEW_APPLE_texture_range = GL_FALSE;
GLboolean __GLEW_APPLE_transform_hint = GL_FALSE;
GLboolean __GLEW_APPLE_vertex_array_object = GL_FALSE;
GLboolean __GLEW_APPLE_vertex_array_range = GL_FALSE;
GLboolean __GLEW_APPLE_vertex_program_evaluators = GL_FALSE;
GLboolean __GLEW_APPLE_ycbcr_422 = GL_FALSE;
GLboolean __GLEW_ARB_ES2_compatibility = GL_FALSE;
GLboolean __GLEW_ARB_ES3_1_compatibility = GL_FALSE;
GLboolean __GLEW_ARB_ES3_2_compatibility = GL_FALSE;
GLboolean __GLEW_ARB_ES3_compatibility = GL_FALSE;
GLboolean __GLEW_ARB_arrays_of_arrays = GL_FALSE;
GLboolean __GLEW_ARB_base_instance = GL_FALSE;
GLboolean __GLEW_ARB_bindless_texture = GL_FALSE;
GLboolean __GLEW_ARB_blend_func_extended = GL_FALSE;
GLboolean __GLEW_ARB_buffer_storage = GL_FALSE;
GLboolean __GLEW_ARB_cl_event = GL_FALSE;
GLboolean __GLEW_ARB_clear_buffer_object = GL_FALSE;
GLboolean __GLEW_ARB_clear_texture = GL_FALSE;
GLboolean __GLEW_ARB_clip_control = GL_FALSE;
GLboolean __GLEW_ARB_color_buffer_float = GL_FALSE;
GLboolean __GLEW_ARB_compatibility = GL_FALSE;
GLboolean __GLEW_ARB_compressed_texture_pixel_storage = GL_FALSE;
GLboolean __GLEW_ARB_compute_shader = GL_FALSE;
GLboolean __GLEW_ARB_compute_variable_group_size = GL_FALSE;
GLboolean __GLEW_ARB_conditional_render_inverted = GL_FALSE;
GLboolean __GLEW_ARB_conservative_depth = GL_FALSE;
GLboolean __GLEW_ARB_copy_buffer = GL_FALSE;
GLboolean __GLEW_ARB_copy_image = GL_FALSE;
GLboolean __GLEW_ARB_cull_distance = GL_FALSE;
GLboolean __GLEW_ARB_debug_output = GL_FALSE;
GLboolean __GLEW_ARB_depth_buffer_float = GL_FALSE;
GLboolean __GLEW_ARB_depth_clamp = GL_FALSE;
GLboolean __GLEW_ARB_depth_texture = GL_FALSE;
GLboolean __GLEW_ARB_derivative_control = GL_FALSE;
GLboolean __GLEW_ARB_direct_state_access = GL_FALSE;
GLboolean __GLEW_ARB_draw_buffers = GL_FALSE;
GLboolean __GLEW_ARB_draw_buffers_blend = GL_FALSE;
GLboolean __GLEW_ARB_draw_elements_base_vertex = GL_FALSE;
GLboolean __GLEW_ARB_draw_indirect = GL_FALSE;
GLboolean __GLEW_ARB_draw_instanced = GL_FALSE;
GLboolean __GLEW_ARB_enhanced_layouts = GL_FALSE;
GLboolean __GLEW_ARB_explicit_attrib_location = GL_FALSE;
GLboolean __GLEW_ARB_explicit_uniform_location = GL_FALSE;
GLboolean __GLEW_ARB_fragment_coord_conventions = GL_FALSE;
GLboolean __GLEW_ARB_fragment_layer_viewport = GL_FALSE;
GLboolean __GLEW_ARB_fragment_program = GL_FALSE;
GLboolean __GLEW_ARB_fragment_program_shadow = GL_FALSE;
GLboolean __GLEW_ARB_fragment_shader = GL_FALSE;
GLboolean __GLEW_ARB_fragment_shader_interlock = GL_FALSE;
GLboolean __GLEW_ARB_framebuffer_no_attachments = GL_FALSE;
GLboolean __GLEW_ARB_framebuffer_object = GL_FALSE;
GLboolean __GLEW_ARB_framebuffer_sRGB = GL_FALSE;
GLboolean __GLEW_ARB_geometry_shader4 = GL_FALSE;
GLboolean __GLEW_ARB_get_program_binary = GL_FALSE;
GLboolean __GLEW_ARB_get_texture_sub_image = GL_FALSE;
GLboolean __GLEW_ARB_gpu_shader5 = GL_FALSE;
GLboolean __GLEW_ARB_gpu_shader_fp64 = GL_FALSE;
GLboolean __GLEW_ARB_gpu_shader_int64 = GL_FALSE;
GLboolean __GLEW_ARB_half_float_pixel = GL_FALSE;
GLboolean __GLEW_ARB_half_float_vertex = GL_FALSE;
GLboolean __GLEW_ARB_imaging = GL_FALSE;
GLboolean __GLEW_ARB_indirect_parameters = GL_FALSE;
GLboolean __GLEW_ARB_instanced_arrays = GL_FALSE;
GLboolean __GLEW_ARB_internalformat_query = GL_FALSE;
GLboolean __GLEW_ARB_internalformat_query2 = GL_FALSE;
GLboolean __GLEW_ARB_invalidate_subdata = GL_FALSE;
GLboolean __GLEW_ARB_map_buffer_alignment = GL_FALSE;
GLboolean __GLEW_ARB_map_buffer_range = GL_FALSE;
GLboolean __GLEW_ARB_matrix_palette = GL_FALSE;
GLboolean __GLEW_ARB_multi_bind = GL_FALSE;
GLboolean __GLEW_ARB_multi_draw_indirect = GL_FALSE;
GLboolean __GLEW_ARB_multisample = GL_FALSE;
GLboolean __GLEW_ARB_multitexture = GL_FALSE;
GLboolean __GLEW_ARB_occlusion_query = GL_FALSE;
GLboolean __GLEW_ARB_occlusion_query2 = GL_FALSE;
GLboolean __GLEW_ARB_parallel_shader_compile = GL_FALSE;
GLboolean __GLEW_ARB_pipeline_statistics_query = GL_FALSE;
GLboolean __GLEW_ARB_pixel_buffer_object = GL_FALSE;
GLboolean __GLEW_ARB_point_parameters = GL_FALSE;
GLboolean __GLEW_ARB_point_sprite = GL_FALSE;
GLboolean __GLEW_ARB_post_depth_coverage = GL_FALSE;
GLboolean __GLEW_ARB_program_interface_query = GL_FALSE;
GLboolean __GLEW_ARB_provoking_vertex = GL_FALSE;
GLboolean __GLEW_ARB_query_buffer_object = GL_FALSE;
GLboolean __GLEW_ARB_robust_buffer_access_behavior = GL_FALSE;
GLboolean __GLEW_ARB_robustness = GL_FALSE;
GLboolean __GLEW_ARB_robustness_application_isolation = GL_FALSE;
GLboolean __GLEW_ARB_robustness_share_group_isolation = GL_FALSE;
GLboolean __GLEW_ARB_sample_locations = GL_FALSE;
GLboolean __GLEW_ARB_sample_shading = GL_FALSE;
GLboolean __GLEW_ARB_sampler_objects = GL_FALSE;
GLboolean __GLEW_ARB_seamless_cube_map = GL_FALSE;
GLboolean __GLEW_ARB_seamless_cubemap_per_texture = GL_FALSE;
GLboolean __GLEW_ARB_separate_shader_objects = GL_FALSE;
GLboolean __GLEW_ARB_shader_atomic_counter_ops = GL_FALSE;
GLboolean __GLEW_ARB_shader_atomic_counters = GL_FALSE;
GLboolean __GLEW_ARB_shader_ballot = GL_FALSE;
GLboolean __GLEW_ARB_shader_bit_encoding = GL_FALSE;
GLboolean __GLEW_ARB_shader_clock = GL_FALSE;
GLboolean __GLEW_ARB_shader_draw_parameters = GL_FALSE;
GLboolean __GLEW_ARB_shader_group_vote = GL_FALSE;
GLboolean __GLEW_ARB_shader_image_load_store = GL_FALSE;
GLboolean __GLEW_ARB_shader_image_size = GL_FALSE;
GLboolean __GLEW_ARB_shader_objects = GL_FALSE;
GLboolean __GLEW_ARB_shader_precision = GL_FALSE;
GLboolean __GLEW_ARB_shader_stencil_export = GL_FALSE;
GLboolean __GLEW_ARB_shader_storage_buffer_object = GL_FALSE;
GLboolean __GLEW_ARB_shader_subroutine = GL_FALSE;
GLboolean __GLEW_ARB_shader_texture_image_samples = GL_FALSE;
GLboolean __GLEW_ARB_shader_texture_lod = GL_FALSE;
GLboolean __GLEW_ARB_shader_viewport_layer_array = GL_FALSE;
GLboolean __GLEW_ARB_shading_language_100 = GL_FALSE;
GLboolean __GLEW_ARB_shading_language_420pack = GL_FALSE;
GLboolean __GLEW_ARB_shading_language_include = GL_FALSE;
GLboolean __GLEW_ARB_shading_language_packing = GL_FALSE;
GLboolean __GLEW_ARB_shadow = GL_FALSE;
GLboolean __GLEW_ARB_shadow_ambient = GL_FALSE;
GLboolean __GLEW_ARB_sparse_buffer = GL_FALSE;
GLboolean __GLEW_ARB_sparse_texture = GL_FALSE;
GLboolean __GLEW_ARB_sparse_texture2 = GL_FALSE;
GLboolean __GLEW_ARB_sparse_texture_clamp = GL_FALSE;
GLboolean __GLEW_ARB_stencil_texturing = GL_FALSE;
GLboolean __GLEW_ARB_sync = GL_FALSE;
GLboolean __GLEW_ARB_tessellation_shader = GL_FALSE;
GLboolean __GLEW_ARB_texture_barrier = GL_FALSE;
GLboolean __GLEW_ARB_texture_border_clamp = GL_FALSE;
GLboolean __GLEW_ARB_texture_buffer_object = GL_FALSE;
GLboolean __GLEW_ARB_texture_buffer_object_rgb32 = GL_FALSE;
GLboolean __GLEW_ARB_texture_buffer_range = GL_FALSE;
GLboolean __GLEW_ARB_texture_compression = GL_FALSE;
GLboolean __GLEW_ARB_texture_compression_bptc = GL_FALSE;
GLboolean __GLEW_ARB_texture_compression_rgtc = GL_FALSE;
GLboolean __GLEW_ARB_texture_cube_map = GL_FALSE;
GLboolean __GLEW_ARB_texture_cube_map_array = GL_FALSE;
GLboolean __GLEW_ARB_texture_env_add = GL_FALSE;
GLboolean __GLEW_ARB_texture_env_combine = GL_FALSE;
GLboolean __GLEW_ARB_texture_env_crossbar = GL_FALSE;
GLboolean __GLEW_ARB_texture_env_dot3 = GL_FALSE;
GLboolean __GLEW_ARB_texture_filter_minmax = GL_FALSE;
GLboolean __GLEW_ARB_texture_float = GL_FALSE;
GLboolean __GLEW_ARB_texture_gather = GL_FALSE;
GLboolean __GLEW_ARB_texture_mirror_clamp_to_edge = GL_FALSE;
GLboolean __GLEW_ARB_texture_mirrored_repeat = GL_FALSE;
GLboolean __GLEW_ARB_texture_multisample = GL_FALSE;
GLboolean __GLEW_ARB_texture_non_power_of_two = GL_FALSE;
GLboolean __GLEW_ARB_texture_query_levels = GL_FALSE;
GLboolean __GLEW_ARB_texture_query_lod = GL_FALSE;
GLboolean __GLEW_ARB_texture_rectangle = GL_FALSE;
GLboolean __GLEW_ARB_texture_rg = GL_FALSE;
GLboolean __GLEW_ARB_texture_rgb10_a2ui = GL_FALSE;
GLboolean __GLEW_ARB_texture_stencil8 = GL_FALSE;
GLboolean __GLEW_ARB_texture_storage = GL_FALSE;
GLboolean __GLEW_ARB_texture_storage_multisample = GL_FALSE;
GLboolean __GLEW_ARB_texture_swizzle = GL_FALSE;
GLboolean __GLEW_ARB_texture_view = GL_FALSE;
GLboolean __GLEW_ARB_timer_query = GL_FALSE;
GLboolean __GLEW_ARB_transform_feedback2 = GL_FALSE;
GLboolean __GLEW_ARB_transform_feedback3 = GL_FALSE;
GLboolean __GLEW_ARB_transform_feedback_instanced = GL_FALSE;
GLboolean __GLEW_ARB_transform_feedback_overflow_query = GL_FALSE;
GLboolean __GLEW_ARB_transpose_matrix = GL_FALSE;
GLboolean __GLEW_ARB_uniform_buffer_object = GL_FALSE;
GLboolean __GLEW_ARB_vertex_array_bgra = GL_FALSE;
GLboolean __GLEW_ARB_vertex_array_object = GL_FALSE;
GLboolean __GLEW_ARB_vertex_attrib_64bit = GL_FALSE;
GLboolean __GLEW_ARB_vertex_attrib_binding = GL_FALSE;
GLboolean __GLEW_ARB_vertex_blend = GL_FALSE;
GLboolean __GLEW_ARB_vertex_buffer_object = GL_FALSE;
GLboolean __GLEW_ARB_vertex_program = GL_FALSE;
GLboolean __GLEW_ARB_vertex_shader = GL_FALSE;
GLboolean __GLEW_ARB_vertex_type_10f_11f_11f_rev = GL_FALSE;
GLboolean __GLEW_ARB_vertex_type_2_10_10_10_rev = GL_FALSE;
GLboolean __GLEW_ARB_viewport_array = GL_FALSE;
GLboolean __GLEW_ARB_window_pos = GL_FALSE;
GLboolean __GLEW_ATIX_point_sprites = GL_FALSE;
GLboolean __GLEW_ATIX_texture_env_combine3 = GL_FALSE;
GLboolean __GLEW_ATIX_texture_env_route = GL_FALSE;
GLboolean __GLEW_ATIX_vertex_shader_output_point_size = GL_FALSE;
GLboolean __GLEW_ATI_draw_buffers = GL_FALSE;
GLboolean __GLEW_ATI_element_array = GL_FALSE;
GLboolean __GLEW_ATI_envmap_bumpmap = GL_FALSE;
GLboolean __GLEW_ATI_fragment_shader = GL_FALSE;
GLboolean __GLEW_ATI_map_object_buffer = GL_FALSE;
GLboolean __GLEW_ATI_meminfo = GL_FALSE;
GLboolean __GLEW_ATI_pn_triangles = GL_FALSE;
GLboolean __GLEW_ATI_separate_stencil = GL_FALSE;
GLboolean __GLEW_ATI_shader_texture_lod = GL_FALSE;
GLboolean __GLEW_ATI_text_fragment_shader = GL_FALSE;
GLboolean __GLEW_ATI_texture_compression_3dc = GL_FALSE;
GLboolean __GLEW_ATI_texture_env_combine3 = GL_FALSE;
GLboolean __GLEW_ATI_texture_float = GL_FALSE;
GLboolean __GLEW_ATI_texture_mirror_once = GL_FALSE;
GLboolean __GLEW_ATI_vertex_array_object = GL_FALSE;
GLboolean __GLEW_ATI_vertex_attrib_array_object = GL_FALSE;
GLboolean __GLEW_ATI_vertex_streams = GL_FALSE;
GLboolean __GLEW_EXT_422_pixels = GL_FALSE;
GLboolean __GLEW_EXT_Cg_shader = GL_FALSE;
GLboolean __GLEW_EXT_abgr = GL_FALSE;
GLboolean __GLEW_EXT_bgra = GL_FALSE;
GLboolean __GLEW_EXT_bindable_uniform = GL_FALSE;
GLboolean __GLEW_EXT_blend_color = GL_FALSE;
GLboolean __GLEW_EXT_blend_equation_separate = GL_FALSE;
GLboolean __GLEW_EXT_blend_func_separate = GL_FALSE;
GLboolean __GLEW_EXT_blend_logic_op = GL_FALSE;
GLboolean __GLEW_EXT_blend_minmax = GL_FALSE;
GLboolean __GLEW_EXT_blend_subtract = GL_FALSE;
GLboolean __GLEW_EXT_clip_volume_hint = GL_FALSE;
GLboolean __GLEW_EXT_cmyka = GL_FALSE;
GLboolean __GLEW_EXT_color_subtable = GL_FALSE;
GLboolean __GLEW_EXT_compiled_vertex_array = GL_FALSE;
GLboolean __GLEW_EXT_convolution = GL_FALSE;
GLboolean __GLEW_EXT_coordinate_frame = GL_FALSE;
GLboolean __GLEW_EXT_copy_texture = GL_FALSE;
GLboolean __GLEW_EXT_cull_vertex = GL_FALSE;
GLboolean __GLEW_EXT_debug_label = GL_FALSE;
GLboolean __GLEW_EXT_debug_marker = GL_FALSE;
GLboolean __GLEW_EXT_depth_bounds_test = GL_FALSE;
GLboolean __GLEW_EXT_direct_state_access = GL_FALSE;
GLboolean __GLEW_EXT_draw_buffers2 = GL_FALSE;
GLboolean __GLEW_EXT_draw_instanced = GL_FALSE;
GLboolean __GLEW_EXT_draw_range_elements = GL_FALSE;
GLboolean __GLEW_EXT_fog_coord = GL_FALSE;
GLboolean __GLEW_EXT_fragment_lighting = GL_FALSE;
GLboolean __GLEW_EXT_framebuffer_blit = GL_FALSE;
GLboolean __GLEW_EXT_framebuffer_multisample = GL_FALSE;
GLboolean __GLEW_EXT_framebuffer_multisample_blit_scaled = GL_FALSE;
GLboolean __GLEW_EXT_framebuffer_object = GL_FALSE;
GLboolean __GLEW_EXT_framebuffer_sRGB = GL_FALSE;
GLboolean __GLEW_EXT_geometry_shader4 = GL_FALSE;
GLboolean __GLEW_EXT_gpu_program_parameters = GL_FALSE;
GLboolean __GLEW_EXT_gpu_shader4 = GL_FALSE;
GLboolean __GLEW_EXT_histogram = GL_FALSE;
GLboolean __GLEW_EXT_index_array_formats = GL_FALSE;
GLboolean __GLEW_EXT_index_func = GL_FALSE;
GLboolean __GLEW_EXT_index_material = GL_FALSE;
GLboolean __GLEW_EXT_index_texture = GL_FALSE;
GLboolean __GLEW_EXT_light_texture = GL_FALSE;
GLboolean __GLEW_EXT_misc_attribute = GL_FALSE;
GLboolean __GLEW_EXT_multi_draw_arrays = GL_FALSE;
GLboolean __GLEW_EXT_multisample = GL_FALSE;
GLboolean __GLEW_EXT_packed_depth_stencil = GL_FALSE;
GLboolean __GLEW_EXT_packed_float = GL_FALSE;
GLboolean __GLEW_EXT_packed_pixels = GL_FALSE;
GLboolean __GLEW_EXT_paletted_texture = GL_FALSE;
GLboolean __GLEW_EXT_pixel_buffer_object = GL_FALSE;
GLboolean __GLEW_EXT_pixel_transform = GL_FALSE;
GLboolean __GLEW_EXT_pixel_transform_color_table = GL_FALSE;
GLboolean __GLEW_EXT_point_parameters = GL_FALSE;
GLboolean __GLEW_EXT_polygon_offset = GL_FALSE;
GLboolean __GLEW_EXT_polygon_offset_clamp = GL_FALSE;
GLboolean __GLEW_EXT_post_depth_coverage = GL_FALSE;
GLboolean __GLEW_EXT_provoking_vertex = GL_FALSE;
GLboolean __GLEW_EXT_raster_multisample = GL_FALSE;
GLboolean __GLEW_EXT_rescale_normal = GL_FALSE;
GLboolean __GLEW_EXT_scene_marker = GL_FALSE;
GLboolean __GLEW_EXT_secondary_color = GL_FALSE;
GLboolean __GLEW_EXT_separate_shader_objects = GL_FALSE;
GLboolean __GLEW_EXT_separate_specular_color = GL_FALSE;
GLboolean __GLEW_EXT_shader_image_load_formatted = GL_FALSE;
GLboolean __GLEW_EXT_shader_image_load_store = GL_FALSE;
GLboolean __GLEW_EXT_shader_integer_mix = GL_FALSE;
GLboolean __GLEW_EXT_shadow_funcs = GL_FALSE;
GLboolean __GLEW_EXT_shared_texture_palette = GL_FALSE;
GLboolean __GLEW_EXT_sparse_texture2 = GL_FALSE;
GLboolean __GLEW_EXT_stencil_clear_tag = GL_FALSE;
GLboolean __GLEW_EXT_stencil_two_side = GL_FALSE;
GLboolean __GLEW_EXT_stencil_wrap = GL_FALSE;
GLboolean __GLEW_EXT_subtexture = GL_FALSE;
GLboolean __GLEW_EXT_texture = GL_FALSE;
GLboolean __GLEW_EXT_texture3D = GL_FALSE;
GLboolean __GLEW_EXT_texture_array = GL_FALSE;
GLboolean __GLEW_EXT_texture_buffer_object = GL_FALSE;
GLboolean __GLEW_EXT_texture_compression_dxt1 = GL_FALSE;
GLboolean __GLEW_EXT_texture_compression_latc = GL_FALSE;
GLboolean __GLEW_EXT_texture_compression_rgtc = GL_FALSE;
GLboolean __GLEW_EXT_texture_compression_s3tc = GL_FALSE;
GLboolean __GLEW_EXT_texture_cube_map = GL_FALSE;
GLboolean __GLEW_EXT_texture_edge_clamp = GL_FALSE;
GLboolean __GLEW_EXT_texture_env = GL_FALSE;
GLboolean __GLEW_EXT_texture_env_add = GL_FALSE;
GLboolean __GLEW_EXT_texture_env_combine = GL_FALSE;
GLboolean __GLEW_EXT_texture_env_dot3 = GL_FALSE;
GLboolean __GLEW_EXT_texture_filter_anisotropic = GL_FALSE;
GLboolean __GLEW_EXT_texture_filter_minmax = GL_FALSE;
GLboolean __GLEW_EXT_texture_integer = GL_FALSE;
GLboolean __GLEW_EXT_texture_lod_bias = GL_FALSE;
GLboolean __GLEW_EXT_texture_mirror_clamp = GL_FALSE;
GLboolean __GLEW_EXT_texture_object = GL_FALSE;
GLboolean __GLEW_EXT_texture_perturb_normal = GL_FALSE;
GLboolean __GLEW_EXT_texture_rectangle = GL_FALSE;
GLboolean __GLEW_EXT_texture_sRGB = GL_FALSE;
GLboolean __GLEW_EXT_texture_sRGB_decode = GL_FALSE;
GLboolean __GLEW_EXT_texture_shared_exponent = GL_FALSE;
GLboolean __GLEW_EXT_texture_snorm = GL_FALSE;
GLboolean __GLEW_EXT_texture_swizzle = GL_FALSE;
GLboolean __GLEW_EXT_timer_query = GL_FALSE;
GLboolean __GLEW_EXT_transform_feedback = GL_FALSE;
GLboolean __GLEW_EXT_vertex_array = GL_FALSE;
GLboolean __GLEW_EXT_vertex_array_bgra = GL_FALSE;
GLboolean __GLEW_EXT_vertex_attrib_64bit = GL_FALSE;
GLboolean __GLEW_EXT_vertex_shader = GL_FALSE;
GLboolean __GLEW_EXT_vertex_weighting = GL_FALSE;
GLboolean __GLEW_EXT_x11_sync_object = GL_FALSE;
GLboolean __GLEW_GREMEDY_frame_terminator = GL_FALSE;
GLboolean __GLEW_GREMEDY_string_marker = GL_FALSE;
GLboolean __GLEW_HP_convolution_border_modes = GL_FALSE;
GLboolean __GLEW_HP_image_transform = GL_FALSE;
GLboolean __GLEW_HP_occlusion_test = GL_FALSE;
GLboolean __GLEW_HP_texture_lighting = GL_FALSE;
GLboolean __GLEW_IBM_cull_vertex = GL_FALSE;
GLboolean __GLEW_IBM_multimode_draw_arrays = GL_FALSE;
GLboolean __GLEW_IBM_rasterpos_clip = GL_FALSE;
GLboolean __GLEW_IBM_static_data = GL_FALSE;
GLboolean __GLEW_IBM_texture_mirrored_repeat = GL_FALSE;
GLboolean __GLEW_IBM_vertex_array_lists = GL_FALSE;
GLboolean __GLEW_INGR_color_clamp = GL_FALSE;
GLboolean __GLEW_INGR_interlace_read = GL_FALSE;
GLboolean __GLEW_INTEL_fragment_shader_ordering = GL_FALSE;
GLboolean __GLEW_INTEL_framebuffer_CMAA = GL_FALSE;
GLboolean __GLEW_INTEL_map_texture = GL_FALSE;
GLboolean __GLEW_INTEL_parallel_arrays = GL_FALSE;
GLboolean __GLEW_INTEL_performance_query = GL_FALSE;
GLboolean __GLEW_INTEL_texture_scissor = GL_FALSE;
GLboolean __GLEW_KHR_blend_equation_advanced = GL_FALSE;
GLboolean __GLEW_KHR_blend_equation_advanced_coherent = GL_FALSE;
GLboolean __GLEW_KHR_context_flush_control = GL_FALSE;
GLboolean __GLEW_KHR_debug = GL_FALSE;
GLboolean __GLEW_KHR_no_error = GL_FALSE;
GLboolean __GLEW_KHR_robust_buffer_access_behavior = GL_FALSE;
GLboolean __GLEW_KHR_robustness = GL_FALSE;
GLboolean __GLEW_KHR_texture_compression_astc_hdr = GL_FALSE;
GLboolean __GLEW_KHR_texture_compression_astc_ldr = GL_FALSE;
GLboolean __GLEW_KTX_buffer_region = GL_FALSE;
GLboolean __GLEW_MESAX_texture_stack = GL_FALSE;
GLboolean __GLEW_MESA_pack_invert = GL_FALSE;
GLboolean __GLEW_MESA_resize_buffers = GL_FALSE;
GLboolean __GLEW_MESA_window_pos = GL_FALSE;
GLboolean __GLEW_MESA_ycbcr_texture = GL_FALSE;
GLboolean __GLEW_NVX_conditional_render = GL_FALSE;
GLboolean __GLEW_NVX_gpu_memory_info = GL_FALSE;
GLboolean __GLEW_NV_bindless_multi_draw_indirect = GL_FALSE;
GLboolean __GLEW_NV_bindless_multi_draw_indirect_count = GL_FALSE;
GLboolean __GLEW_NV_bindless_texture = GL_FALSE;
GLboolean __GLEW_NV_blend_equation_advanced = GL_FALSE;
GLboolean __GLEW_NV_blend_equation_advanced_coherent = GL_FALSE;
GLboolean __GLEW_NV_blend_square = GL_FALSE;
GLboolean __GLEW_NV_compute_program5 = GL_FALSE;
GLboolean __GLEW_NV_conditional_render = GL_FALSE;
GLboolean __GLEW_NV_conservative_raster = GL_FALSE;
GLboolean __GLEW_NV_conservative_raster_dilate = GL_FALSE;
GLboolean __GLEW_NV_copy_depth_to_color = GL_FALSE;
GLboolean __GLEW_NV_copy_image = GL_FALSE;
GLboolean __GLEW_NV_deep_texture3D = GL_FALSE;
GLboolean __GLEW_NV_depth_buffer_float = GL_FALSE;
GLboolean __GLEW_NV_depth_clamp = GL_FALSE;
GLboolean __GLEW_NV_depth_range_unclamped = GL_FALSE;
GLboolean __GLEW_NV_draw_texture = GL_FALSE;
GLboolean __GLEW_NV_evaluators = GL_FALSE;
GLboolean __GLEW_NV_explicit_multisample = GL_FALSE;
GLboolean __GLEW_NV_fence = GL_FALSE;
GLboolean __GLEW_NV_fill_rectangle = GL_FALSE;
GLboolean __GLEW_NV_float_buffer = GL_FALSE;
GLboolean __GLEW_NV_fog_distance = GL_FALSE;
GLboolean __GLEW_NV_fragment_coverage_to_color = GL_FALSE;
GLboolean __GLEW_NV_fragment_program = GL_FALSE;
GLboolean __GLEW_NV_fragment_program2 = GL_FALSE;
GLboolean __GLEW_NV_fragment_program4 = GL_FALSE;
GLboolean __GLEW_NV_fragment_program_option = GL_FALSE;
GLboolean __GLEW_NV_fragment_shader_interlock = GL_FALSE;
GLboolean __GLEW_NV_framebuffer_mixed_samples = GL_FALSE;
GLboolean __GLEW_NV_framebuffer_multisample_coverage = GL_FALSE;
GLboolean __GLEW_NV_geometry_program4 = GL_FALSE;
GLboolean __GLEW_NV_geometry_shader4 = GL_FALSE;
GLboolean __GLEW_NV_geometry_shader_passthrough = GL_FALSE;
GLboolean __GLEW_NV_gpu_program4 = GL_FALSE;
GLboolean __GLEW_NV_gpu_program5 = GL_FALSE;
GLboolean __GLEW_NV_gpu_program5_mem_extended = GL_FALSE;
GLboolean __GLEW_NV_gpu_program_fp64 = GL_FALSE;
GLboolean __GLEW_NV_gpu_shader5 = GL_FALSE;
GLboolean __GLEW_NV_half_float = GL_FALSE;
GLboolean __GLEW_NV_internalformat_sample_query = GL_FALSE;
GLboolean __GLEW_NV_light_max_exponent = GL_FALSE;
GLboolean __GLEW_NV_multisample_coverage = GL_FALSE;
GLboolean __GLEW_NV_multisample_filter_hint = GL_FALSE;
GLboolean __GLEW_NV_occlusion_query = GL_FALSE;
GLboolean __GLEW_NV_packed_depth_stencil = GL_FALSE;
GLboolean __GLEW_NV_parameter_buffer_object = GL_FALSE;
GLboolean __GLEW_NV_parameter_buffer_object2 = GL_FALSE;
GLboolean __GLEW_NV_path_rendering = GL_FALSE;
GLboolean __GLEW_NV_path_rendering_shared_edge = GL_FALSE;
GLboolean __GLEW_NV_pixel_data_range = GL_FALSE;
GLboolean __GLEW_NV_point_sprite = GL_FALSE;
GLboolean __GLEW_NV_present_video = GL_FALSE;
GLboolean __GLEW_NV_primitive_restart = GL_FALSE;
GLboolean __GLEW_NV_register_combiners = GL_FALSE;
GLboolean __GLEW_NV_register_combiners2 = GL_FALSE;
GLboolean __GLEW_NV_sample_locations = GL_FALSE;
GLboolean __GLEW_NV_sample_mask_override_coverage = GL_FALSE;
GLboolean __GLEW_NV_shader_atomic_counters = GL_FALSE;
GLboolean __GLEW_NV_shader_atomic_float = GL_FALSE;
GLboolean __GLEW_NV_shader_atomic_fp16_vector = GL_FALSE;
GLboolean __GLEW_NV_shader_atomic_int64 = GL_FALSE;
GLboolean __GLEW_NV_shader_buffer_load = GL_FALSE;
GLboolean __GLEW_NV_shader_storage_buffer_object = GL_FALSE;
GLboolean __GLEW_NV_shader_thread_group = GL_FALSE;
GLboolean __GLEW_NV_shader_thread_shuffle = GL_FALSE;
GLboolean __GLEW_NV_tessellation_program5 = GL_FALSE;
GLboolean __GLEW_NV_texgen_emboss = GL_FALSE;
GLboolean __GLEW_NV_texgen_reflection = GL_FALSE;
GLboolean __GLEW_NV_texture_barrier = GL_FALSE;
GLboolean __GLEW_NV_texture_compression_vtc = GL_FALSE;
GLboolean __GLEW_NV_texture_env_combine4 = GL_FALSE;
GLboolean __GLEW_NV_texture_expand_normal = GL_FALSE;
GLboolean __GLEW_NV_texture_multisample = GL_FALSE;
GLboolean __GLEW_NV_texture_rectangle = GL_FALSE;
GLboolean __GLEW_NV_texture_shader = GL_FALSE;
GLboolean __GLEW_NV_texture_shader2 = GL_FALSE;
GLboolean __GLEW_NV_texture_shader3 = GL_FALSE;
GLboolean __GLEW_NV_transform_feedback = GL_FALSE;
GLboolean __GLEW_NV_transform_feedback2 = GL_FALSE;
GLboolean __GLEW_NV_uniform_buffer_unified_memory = GL_FALSE;
GLboolean __GLEW_NV_vdpau_interop = GL_FALSE;
GLboolean __GLEW_NV_vertex_array_range = GL_FALSE;
GLboolean __GLEW_NV_vertex_array_range2 = GL_FALSE;
GLboolean __GLEW_NV_vertex_attrib_integer_64bit = GL_FALSE;
GLboolean __GLEW_NV_vertex_buffer_unified_memory = GL_FALSE;
GLboolean __GLEW_NV_vertex_program = GL_FALSE;
GLboolean __GLEW_NV_vertex_program1_1 = GL_FALSE;
GLboolean __GLEW_NV_vertex_program2 = GL_FALSE;
GLboolean __GLEW_NV_vertex_program2_option = GL_FALSE;
GLboolean __GLEW_NV_vertex_program3 = GL_FALSE;
GLboolean __GLEW_NV_vertex_program4 = GL_FALSE;
GLboolean __GLEW_NV_video_capture = GL_FALSE;
GLboolean __GLEW_NV_viewport_array2 = GL_FALSE;
GLboolean __GLEW_OES_byte_coordinates = GL_FALSE;
GLboolean __GLEW_OES_compressed_paletted_texture = GL_FALSE;
GLboolean __GLEW_OES_read_format = GL_FALSE;
GLboolean __GLEW_OES_single_precision = GL_FALSE;
GLboolean __GLEW_OML_interlace = GL_FALSE;
GLboolean __GLEW_OML_resample = GL_FALSE;
GLboolean __GLEW_OML_subsample = GL_FALSE;
GLboolean __GLEW_OVR_multiview = GL_FALSE;
GLboolean __GLEW_OVR_multiview2 = GL_FALSE;
GLboolean __GLEW_PGI_misc_hints = GL_FALSE;
GLboolean __GLEW_PGI_vertex_hints = GL_FALSE;
GLboolean __GLEW_REGAL_ES1_0_compatibility = GL_FALSE;
GLboolean __GLEW_REGAL_ES1_1_compatibility = GL_FALSE;
GLboolean __GLEW_REGAL_enable = GL_FALSE;
GLboolean __GLEW_REGAL_error_string = GL_FALSE;
GLboolean __GLEW_REGAL_extension_query = GL_FALSE;
GLboolean __GLEW_REGAL_log = GL_FALSE;
GLboolean __GLEW_REGAL_proc_address = GL_FALSE;
GLboolean __GLEW_REND_screen_coordinates = GL_FALSE;
GLboolean __GLEW_S3_s3tc = GL_FALSE;
GLboolean __GLEW_SGIS_color_range = GL_FALSE;
GLboolean __GLEW_SGIS_detail_texture = GL_FALSE;
GLboolean __GLEW_SGIS_fog_function = GL_FALSE;
GLboolean __GLEW_SGIS_generate_mipmap = GL_FALSE;
GLboolean __GLEW_SGIS_multisample = GL_FALSE;
GLboolean __GLEW_SGIS_pixel_texture = GL_FALSE;
GLboolean __GLEW_SGIS_point_line_texgen = GL_FALSE;
GLboolean __GLEW_SGIS_sharpen_texture = GL_FALSE;
GLboolean __GLEW_SGIS_texture4D = GL_FALSE;
GLboolean __GLEW_SGIS_texture_border_clamp = GL_FALSE;
GLboolean __GLEW_SGIS_texture_edge_clamp = GL_FALSE;
GLboolean __GLEW_SGIS_texture_filter4 = GL_FALSE;
GLboolean __GLEW_SGIS_texture_lod = GL_FALSE;
GLboolean __GLEW_SGIS_texture_select = GL_FALSE;
GLboolean __GLEW_SGIX_async = GL_FALSE;
GLboolean __GLEW_SGIX_async_histogram = GL_FALSE;
GLboolean __GLEW_SGIX_async_pixel = GL_FALSE;
GLboolean __GLEW_SGIX_blend_alpha_minmax = GL_FALSE;
GLboolean __GLEW_SGIX_clipmap = GL_FALSE;
GLboolean __GLEW_SGIX_convolution_accuracy = GL_FALSE;
GLboolean __GLEW_SGIX_depth_texture = GL_FALSE;
GLboolean __GLEW_SGIX_flush_raster = GL_FALSE;
GLboolean __GLEW_SGIX_fog_offset = GL_FALSE;
GLboolean __GLEW_SGIX_fog_texture = GL_FALSE;
GLboolean __GLEW_SGIX_fragment_specular_lighting = GL_FALSE;
GLboolean __GLEW_SGIX_framezoom = GL_FALSE;
GLboolean __GLEW_SGIX_interlace = GL_FALSE;
GLboolean __GLEW_SGIX_ir_instrument1 = GL_FALSE;
GLboolean __GLEW_SGIX_list_priority = GL_FALSE;
GLboolean __GLEW_SGIX_pixel_texture = GL_FALSE;
GLboolean __GLEW_SGIX_pixel_texture_bits = GL_FALSE;
GLboolean __GLEW_SGIX_reference_plane = GL_FALSE;
GLboolean __GLEW_SGIX_resample = GL_FALSE;
GLboolean __GLEW_SGIX_shadow = GL_FALSE;
GLboolean __GLEW_SGIX_shadow_ambient = GL_FALSE;
GLboolean __GLEW_SGIX_sprite = GL_FALSE;
GLboolean __GLEW_SGIX_tag_sample_buffer = GL_FALSE;
GLboolean __GLEW_SGIX_texture_add_env = GL_FALSE;
GLboolean __GLEW_SGIX_texture_coordinate_clamp = GL_FALSE;
GLboolean __GLEW_SGIX_texture_lod_bias = GL_FALSE;
GLboolean __GLEW_SGIX_texture_multi_buffer = GL_FALSE;
GLboolean __GLEW_SGIX_texture_range = GL_FALSE;
GLboolean __GLEW_SGIX_texture_scale_bias = GL_FALSE;
GLboolean __GLEW_SGIX_vertex_preclip = GL_FALSE;
GLboolean __GLEW_SGIX_vertex_preclip_hint = GL_FALSE;
GLboolean __GLEW_SGIX_ycrcb = GL_FALSE;
GLboolean __GLEW_SGI_color_matrix = GL_FALSE;
GLboolean __GLEW_SGI_color_table = GL_FALSE;
GLboolean __GLEW_SGI_texture_color_table = GL_FALSE;
GLboolean __GLEW_SUNX_constant_data = GL_FALSE;
GLboolean __GLEW_SUN_convolution_border_modes = GL_FALSE;
GLboolean __GLEW_SUN_global_alpha = GL_FALSE;
GLboolean __GLEW_SUN_mesh_array = GL_FALSE;
GLboolean __GLEW_SUN_read_video_pixels = GL_FALSE;
GLboolean __GLEW_SUN_slice_accum = GL_FALSE;
GLboolean __GLEW_SUN_triangle_list = GL_FALSE;
GLboolean __GLEW_SUN_vertex = GL_FALSE;
GLboolean __GLEW_WIN_phong_shading = GL_FALSE;
GLboolean __GLEW_WIN_specular_fog = GL_FALSE;
GLboolean __GLEW_WIN_swap_hint = GL_FALSE;

#endif /* !GLEW_MX */

#ifdef GL_VERSION_1_2

static GLboolean _glewInit_GL_VERSION_1_2 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCopyTexSubImage3D = (PFNGLCOPYTEXSUBIMAGE3DPROC)glewGetProcAddress((const GLubyte*)"glCopyTexSubImage3D")) == NULL) || r;
  r = ((glDrawRangeElements = (PFNGLDRAWRANGEELEMENTSPROC)glewGetProcAddress((const GLubyte*)"glDrawRangeElements")) == NULL) || r;
  r = ((glTexImage3D = (PFNGLTEXIMAGE3DPROC)glewGetProcAddress((const GLubyte*)"glTexImage3D")) == NULL) || r;
  r = ((glTexSubImage3D = (PFNGLTEXSUBIMAGE3DPROC)glewGetProcAddress((const GLubyte*)"glTexSubImage3D")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_1_2 */

#ifdef GL_VERSION_1_3

static GLboolean _glewInit_GL_VERSION_1_3 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glActiveTexture = (PFNGLACTIVETEXTUREPROC)glewGetProcAddress((const GLubyte*)"glActiveTexture")) == NULL) || r;
  r = ((glClientActiveTexture = (PFNGLCLIENTACTIVETEXTUREPROC)glewGetProcAddress((const GLubyte*)"glClientActiveTexture")) == NULL) || r;
  r = ((glCompressedTexImage1D = (PFNGLCOMPRESSEDTEXIMAGE1DPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexImage1D")) == NULL) || r;
  r = ((glCompressedTexImage2D = (PFNGLCOMPRESSEDTEXIMAGE2DPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexImage2D")) == NULL) || r;
  r = ((glCompressedTexImage3D = (PFNGLCOMPRESSEDTEXIMAGE3DPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexImage3D")) == NULL) || r;
  r = ((glCompressedTexSubImage1D = (PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexSubImage1D")) == NULL) || r;
  r = ((glCompressedTexSubImage2D = (PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexSubImage2D")) == NULL) || r;
  r = ((glCompressedTexSubImage3D = (PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexSubImage3D")) == NULL) || r;
  r = ((glGetCompressedTexImage = (PFNGLGETCOMPRESSEDTEXIMAGEPROC)glewGetProcAddress((const GLubyte*)"glGetCompressedTexImage")) == NULL) || r;
  r = ((glLoadTransposeMatrixd = (PFNGLLOADTRANSPOSEMATRIXDPROC)glewGetProcAddress((const GLubyte*)"glLoadTransposeMatrixd")) == NULL) || r;
  r = ((glLoadTransposeMatrixf = (PFNGLLOADTRANSPOSEMATRIXFPROC)glewGetProcAddress((const GLubyte*)"glLoadTransposeMatrixf")) == NULL) || r;
  r = ((glMultTransposeMatrixd = (PFNGLMULTTRANSPOSEMATRIXDPROC)glewGetProcAddress((const GLubyte*)"glMultTransposeMatrixd")) == NULL) || r;
  r = ((glMultTransposeMatrixf = (PFNGLMULTTRANSPOSEMATRIXFPROC)glewGetProcAddress((const GLubyte*)"glMultTransposeMatrixf")) == NULL) || r;
  r = ((glMultiTexCoord1d = (PFNGLMULTITEXCOORD1DPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1d")) == NULL) || r;
  r = ((glMultiTexCoord1dv = (PFNGLMULTITEXCOORD1DVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1dv")) == NULL) || r;
  r = ((glMultiTexCoord1f = (PFNGLMULTITEXCOORD1FPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1f")) == NULL) || r;
  r = ((glMultiTexCoord1fv = (PFNGLMULTITEXCOORD1FVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1fv")) == NULL) || r;
  r = ((glMultiTexCoord1i = (PFNGLMULTITEXCOORD1IPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1i")) == NULL) || r;
  r = ((glMultiTexCoord1iv = (PFNGLMULTITEXCOORD1IVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1iv")) == NULL) || r;
  r = ((glMultiTexCoord1s = (PFNGLMULTITEXCOORD1SPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1s")) == NULL) || r;
  r = ((glMultiTexCoord1sv = (PFNGLMULTITEXCOORD1SVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1sv")) == NULL) || r;
  r = ((glMultiTexCoord2d = (PFNGLMULTITEXCOORD2DPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2d")) == NULL) || r;
  r = ((glMultiTexCoord2dv = (PFNGLMULTITEXCOORD2DVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2dv")) == NULL) || r;
  r = ((glMultiTexCoord2f = (PFNGLMULTITEXCOORD2FPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2f")) == NULL) || r;
  r = ((glMultiTexCoord2fv = (PFNGLMULTITEXCOORD2FVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2fv")) == NULL) || r;
  r = ((glMultiTexCoord2i = (PFNGLMULTITEXCOORD2IPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2i")) == NULL) || r;
  r = ((glMultiTexCoord2iv = (PFNGLMULTITEXCOORD2IVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2iv")) == NULL) || r;
  r = ((glMultiTexCoord2s = (PFNGLMULTITEXCOORD2SPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2s")) == NULL) || r;
  r = ((glMultiTexCoord2sv = (PFNGLMULTITEXCOORD2SVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2sv")) == NULL) || r;
  r = ((glMultiTexCoord3d = (PFNGLMULTITEXCOORD3DPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3d")) == NULL) || r;
  r = ((glMultiTexCoord3dv = (PFNGLMULTITEXCOORD3DVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3dv")) == NULL) || r;
  r = ((glMultiTexCoord3f = (PFNGLMULTITEXCOORD3FPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3f")) == NULL) || r;
  r = ((glMultiTexCoord3fv = (PFNGLMULTITEXCOORD3FVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3fv")) == NULL) || r;
  r = ((glMultiTexCoord3i = (PFNGLMULTITEXCOORD3IPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3i")) == NULL) || r;
  r = ((glMultiTexCoord3iv = (PFNGLMULTITEXCOORD3IVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3iv")) == NULL) || r;
  r = ((glMultiTexCoord3s = (PFNGLMULTITEXCOORD3SPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3s")) == NULL) || r;
  r = ((glMultiTexCoord3sv = (PFNGLMULTITEXCOORD3SVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3sv")) == NULL) || r;
  r = ((glMultiTexCoord4d = (PFNGLMULTITEXCOORD4DPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4d")) == NULL) || r;
  r = ((glMultiTexCoord4dv = (PFNGLMULTITEXCOORD4DVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4dv")) == NULL) || r;
  r = ((glMultiTexCoord4f = (PFNGLMULTITEXCOORD4FPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4f")) == NULL) || r;
  r = ((glMultiTexCoord4fv = (PFNGLMULTITEXCOORD4FVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4fv")) == NULL) || r;
  r = ((glMultiTexCoord4i = (PFNGLMULTITEXCOORD4IPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4i")) == NULL) || r;
  r = ((glMultiTexCoord4iv = (PFNGLMULTITEXCOORD4IVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4iv")) == NULL) || r;
  r = ((glMultiTexCoord4s = (PFNGLMULTITEXCOORD4SPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4s")) == NULL) || r;
  r = ((glMultiTexCoord4sv = (PFNGLMULTITEXCOORD4SVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4sv")) == NULL) || r;
  r = ((glSampleCoverage = (PFNGLSAMPLECOVERAGEPROC)glewGetProcAddress((const GLubyte*)"glSampleCoverage")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_1_3 */

#ifdef GL_VERSION_1_4

static GLboolean _glewInit_GL_VERSION_1_4 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendColor = (PFNGLBLENDCOLORPROC)glewGetProcAddress((const GLubyte*)"glBlendColor")) == NULL) || r;
  r = ((glBlendEquation = (PFNGLBLENDEQUATIONPROC)glewGetProcAddress((const GLubyte*)"glBlendEquation")) == NULL) || r;
  r = ((glBlendFuncSeparate = (PFNGLBLENDFUNCSEPARATEPROC)glewGetProcAddress((const GLubyte*)"glBlendFuncSeparate")) == NULL) || r;
  r = ((glFogCoordPointer = (PFNGLFOGCOORDPOINTERPROC)glewGetProcAddress((const GLubyte*)"glFogCoordPointer")) == NULL) || r;
  r = ((glFogCoordd = (PFNGLFOGCOORDDPROC)glewGetProcAddress((const GLubyte*)"glFogCoordd")) == NULL) || r;
  r = ((glFogCoorddv = (PFNGLFOGCOORDDVPROC)glewGetProcAddress((const GLubyte*)"glFogCoorddv")) == NULL) || r;
  r = ((glFogCoordf = (PFNGLFOGCOORDFPROC)glewGetProcAddress((const GLubyte*)"glFogCoordf")) == NULL) || r;
  r = ((glFogCoordfv = (PFNGLFOGCOORDFVPROC)glewGetProcAddress((const GLubyte*)"glFogCoordfv")) == NULL) || r;
  r = ((glMultiDrawArrays = (PFNGLMULTIDRAWARRAYSPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawArrays")) == NULL) || r;
  r = ((glMultiDrawElements = (PFNGLMULTIDRAWELEMENTSPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawElements")) == NULL) || r;
  r = ((glPointParameterf = (PFNGLPOINTPARAMETERFPROC)glewGetProcAddress((const GLubyte*)"glPointParameterf")) == NULL) || r;
  r = ((glPointParameterfv = (PFNGLPOINTPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glPointParameterfv")) == NULL) || r;
  r = ((glPointParameteri = (PFNGLPOINTPARAMETERIPROC)glewGetProcAddress((const GLubyte*)"glPointParameteri")) == NULL) || r;
  r = ((glPointParameteriv = (PFNGLPOINTPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glPointParameteriv")) == NULL) || r;
  r = ((glSecondaryColor3b = (PFNGLSECONDARYCOLOR3BPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3b")) == NULL) || r;
  r = ((glSecondaryColor3bv = (PFNGLSECONDARYCOLOR3BVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3bv")) == NULL) || r;
  r = ((glSecondaryColor3d = (PFNGLSECONDARYCOLOR3DPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3d")) == NULL) || r;
  r = ((glSecondaryColor3dv = (PFNGLSECONDARYCOLOR3DVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3dv")) == NULL) || r;
  r = ((glSecondaryColor3f = (PFNGLSECONDARYCOLOR3FPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3f")) == NULL) || r;
  r = ((glSecondaryColor3fv = (PFNGLSECONDARYCOLOR3FVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3fv")) == NULL) || r;
  r = ((glSecondaryColor3i = (PFNGLSECONDARYCOLOR3IPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3i")) == NULL) || r;
  r = ((glSecondaryColor3iv = (PFNGLSECONDARYCOLOR3IVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3iv")) == NULL) || r;
  r = ((glSecondaryColor3s = (PFNGLSECONDARYCOLOR3SPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3s")) == NULL) || r;
  r = ((glSecondaryColor3sv = (PFNGLSECONDARYCOLOR3SVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3sv")) == NULL) || r;
  r = ((glSecondaryColor3ub = (PFNGLSECONDARYCOLOR3UBPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3ub")) == NULL) || r;
  r = ((glSecondaryColor3ubv = (PFNGLSECONDARYCOLOR3UBVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3ubv")) == NULL) || r;
  r = ((glSecondaryColor3ui = (PFNGLSECONDARYCOLOR3UIPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3ui")) == NULL) || r;
  r = ((glSecondaryColor3uiv = (PFNGLSECONDARYCOLOR3UIVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3uiv")) == NULL) || r;
  r = ((glSecondaryColor3us = (PFNGLSECONDARYCOLOR3USPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3us")) == NULL) || r;
  r = ((glSecondaryColor3usv = (PFNGLSECONDARYCOLOR3USVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3usv")) == NULL) || r;
  r = ((glSecondaryColorPointer = (PFNGLSECONDARYCOLORPOINTERPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColorPointer")) == NULL) || r;
  r = ((glWindowPos2d = (PFNGLWINDOWPOS2DPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2d")) == NULL) || r;
  r = ((glWindowPos2dv = (PFNGLWINDOWPOS2DVPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2dv")) == NULL) || r;
  r = ((glWindowPos2f = (PFNGLWINDOWPOS2FPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2f")) == NULL) || r;
  r = ((glWindowPos2fv = (PFNGLWINDOWPOS2FVPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2fv")) == NULL) || r;
  r = ((glWindowPos2i = (PFNGLWINDOWPOS2IPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2i")) == NULL) || r;
  r = ((glWindowPos2iv = (PFNGLWINDOWPOS2IVPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2iv")) == NULL) || r;
  r = ((glWindowPos2s = (PFNGLWINDOWPOS2SPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2s")) == NULL) || r;
  r = ((glWindowPos2sv = (PFNGLWINDOWPOS2SVPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2sv")) == NULL) || r;
  r = ((glWindowPos3d = (PFNGLWINDOWPOS3DPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3d")) == NULL) || r;
  r = ((glWindowPos3dv = (PFNGLWINDOWPOS3DVPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3dv")) == NULL) || r;
  r = ((glWindowPos3f = (PFNGLWINDOWPOS3FPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3f")) == NULL) || r;
  r = ((glWindowPos3fv = (PFNGLWINDOWPOS3FVPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3fv")) == NULL) || r;
  r = ((glWindowPos3i = (PFNGLWINDOWPOS3IPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3i")) == NULL) || r;
  r = ((glWindowPos3iv = (PFNGLWINDOWPOS3IVPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3iv")) == NULL) || r;
  r = ((glWindowPos3s = (PFNGLWINDOWPOS3SPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3s")) == NULL) || r;
  r = ((glWindowPos3sv = (PFNGLWINDOWPOS3SVPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3sv")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_1_4 */

#ifdef GL_VERSION_1_5

static GLboolean _glewInit_GL_VERSION_1_5 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginQuery = (PFNGLBEGINQUERYPROC)glewGetProcAddress((const GLubyte*)"glBeginQuery")) == NULL) || r;
  r = ((glBindBuffer = (PFNGLBINDBUFFERPROC)glewGetProcAddress((const GLubyte*)"glBindBuffer")) == NULL) || r;
  r = ((glBufferData = (PFNGLBUFFERDATAPROC)glewGetProcAddress((const GLubyte*)"glBufferData")) == NULL) || r;
  r = ((glBufferSubData = (PFNGLBUFFERSUBDATAPROC)glewGetProcAddress((const GLubyte*)"glBufferSubData")) == NULL) || r;
  r = ((glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glDeleteBuffers")) == NULL) || r;
  r = ((glDeleteQueries = (PFNGLDELETEQUERIESPROC)glewGetProcAddress((const GLubyte*)"glDeleteQueries")) == NULL) || r;
  r = ((glEndQuery = (PFNGLENDQUERYPROC)glewGetProcAddress((const GLubyte*)"glEndQuery")) == NULL) || r;
  r = ((glGenBuffers = (PFNGLGENBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glGenBuffers")) == NULL) || r;
  r = ((glGenQueries = (PFNGLGENQUERIESPROC)glewGetProcAddress((const GLubyte*)"glGenQueries")) == NULL) || r;
  r = ((glGetBufferParameteriv = (PFNGLGETBUFFERPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetBufferParameteriv")) == NULL) || r;
  r = ((glGetBufferPointerv = (PFNGLGETBUFFERPOINTERVPROC)glewGetProcAddress((const GLubyte*)"glGetBufferPointerv")) == NULL) || r;
  r = ((glGetBufferSubData = (PFNGLGETBUFFERSUBDATAPROC)glewGetProcAddress((const GLubyte*)"glGetBufferSubData")) == NULL) || r;
  r = ((glGetQueryObjectiv = (PFNGLGETQUERYOBJECTIVPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjectiv")) == NULL) || r;
  r = ((glGetQueryObjectuiv = (PFNGLGETQUERYOBJECTUIVPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjectuiv")) == NULL) || r;
  r = ((glGetQueryiv = (PFNGLGETQUERYIVPROC)glewGetProcAddress((const GLubyte*)"glGetQueryiv")) == NULL) || r;
  r = ((glIsBuffer = (PFNGLISBUFFERPROC)glewGetProcAddress((const GLubyte*)"glIsBuffer")) == NULL) || r;
  r = ((glIsQuery = (PFNGLISQUERYPROC)glewGetProcAddress((const GLubyte*)"glIsQuery")) == NULL) || r;
  r = ((glMapBuffer = (PFNGLMAPBUFFERPROC)glewGetProcAddress((const GLubyte*)"glMapBuffer")) == NULL) || r;
  r = ((glUnmapBuffer = (PFNGLUNMAPBUFFERPROC)glewGetProcAddress((const GLubyte*)"glUnmapBuffer")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_1_5 */

#ifdef GL_VERSION_2_0

static GLboolean _glewInit_GL_VERSION_2_0 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glAttachShader = (PFNGLATTACHSHADERPROC)glewGetProcAddress((const GLubyte*)"glAttachShader")) == NULL) || r;
  r = ((glBindAttribLocation = (PFNGLBINDATTRIBLOCATIONPROC)glewGetProcAddress((const GLubyte*)"glBindAttribLocation")) == NULL) || r;
  r = ((glBlendEquationSeparate = (PFNGLBLENDEQUATIONSEPARATEPROC)glewGetProcAddress((const GLubyte*)"glBlendEquationSeparate")) == NULL) || r;
  r = ((glCompileShader = (PFNGLCOMPILESHADERPROC)glewGetProcAddress((const GLubyte*)"glCompileShader")) == NULL) || r;
  r = ((glCreateProgram = (PFNGLCREATEPROGRAMPROC)glewGetProcAddress((const GLubyte*)"glCreateProgram")) == NULL) || r;
  r = ((glCreateShader = (PFNGLCREATESHADERPROC)glewGetProcAddress((const GLubyte*)"glCreateShader")) == NULL) || r;
  r = ((glDeleteProgram = (PFNGLDELETEPROGRAMPROC)glewGetProcAddress((const GLubyte*)"glDeleteProgram")) == NULL) || r;
  r = ((glDeleteShader = (PFNGLDELETESHADERPROC)glewGetProcAddress((const GLubyte*)"glDeleteShader")) == NULL) || r;
  r = ((glDetachShader = (PFNGLDETACHSHADERPROC)glewGetProcAddress((const GLubyte*)"glDetachShader")) == NULL) || r;
  r = ((glDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYPROC)glewGetProcAddress((const GLubyte*)"glDisableVertexAttribArray")) == NULL) || r;
  r = ((glDrawBuffers = (PFNGLDRAWBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glDrawBuffers")) == NULL) || r;
  r = ((glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)glewGetProcAddress((const GLubyte*)"glEnableVertexAttribArray")) == NULL) || r;
  r = ((glGetActiveAttrib = (PFNGLGETACTIVEATTRIBPROC)glewGetProcAddress((const GLubyte*)"glGetActiveAttrib")) == NULL) || r;
  r = ((glGetActiveUniform = (PFNGLGETACTIVEUNIFORMPROC)glewGetProcAddress((const GLubyte*)"glGetActiveUniform")) == NULL) || r;
  r = ((glGetAttachedShaders = (PFNGLGETATTACHEDSHADERSPROC)glewGetProcAddress((const GLubyte*)"glGetAttachedShaders")) == NULL) || r;
  r = ((glGetAttribLocation = (PFNGLGETATTRIBLOCATIONPROC)glewGetProcAddress((const GLubyte*)"glGetAttribLocation")) == NULL) || r;
  r = ((glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)glewGetProcAddress((const GLubyte*)"glGetProgramInfoLog")) == NULL) || r;
  r = ((glGetProgramiv = (PFNGLGETPROGRAMIVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramiv")) == NULL) || r;
  r = ((glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)glewGetProcAddress((const GLubyte*)"glGetShaderInfoLog")) == NULL) || r;
  r = ((glGetShaderSource = (PFNGLGETSHADERSOURCEPROC)glewGetProcAddress((const GLubyte*)"glGetShaderSource")) == NULL) || r;
  r = ((glGetShaderiv = (PFNGLGETSHADERIVPROC)glewGetProcAddress((const GLubyte*)"glGetShaderiv")) == NULL) || r;
  r = ((glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)glewGetProcAddress((const GLubyte*)"glGetUniformLocation")) == NULL) || r;
  r = ((glGetUniformfv = (PFNGLGETUNIFORMFVPROC)glewGetProcAddress((const GLubyte*)"glGetUniformfv")) == NULL) || r;
  r = ((glGetUniformiv = (PFNGLGETUNIFORMIVPROC)glewGetProcAddress((const GLubyte*)"glGetUniformiv")) == NULL) || r;
  r = ((glGetVertexAttribPointerv = (PFNGLGETVERTEXATTRIBPOINTERVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribPointerv")) == NULL) || r;
  r = ((glGetVertexAttribdv = (PFNGLGETVERTEXATTRIBDVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribdv")) == NULL) || r;
  r = ((glGetVertexAttribfv = (PFNGLGETVERTEXATTRIBFVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribfv")) == NULL) || r;
  r = ((glGetVertexAttribiv = (PFNGLGETVERTEXATTRIBIVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribiv")) == NULL) || r;
  r = ((glIsProgram = (PFNGLISPROGRAMPROC)glewGetProcAddress((const GLubyte*)"glIsProgram")) == NULL) || r;
  r = ((glIsShader = (PFNGLISSHADERPROC)glewGetProcAddress((const GLubyte*)"glIsShader")) == NULL) || r;
  r = ((glLinkProgram = (PFNGLLINKPROGRAMPROC)glewGetProcAddress((const GLubyte*)"glLinkProgram")) == NULL) || r;
  r = ((glShaderSource = (PFNGLSHADERSOURCEPROC)glewGetProcAddress((const GLubyte*)"glShaderSource")) == NULL) || r;
  r = ((glStencilFuncSeparate = (PFNGLSTENCILFUNCSEPARATEPROC)glewGetProcAddress((const GLubyte*)"glStencilFuncSeparate")) == NULL) || r;
  r = ((glStencilMaskSeparate = (PFNGLSTENCILMASKSEPARATEPROC)glewGetProcAddress((const GLubyte*)"glStencilMaskSeparate")) == NULL) || r;
  r = ((glStencilOpSeparate = (PFNGLSTENCILOPSEPARATEPROC)glewGetProcAddress((const GLubyte*)"glStencilOpSeparate")) == NULL) || r;
  r = ((glUniform1f = (PFNGLUNIFORM1FPROC)glewGetProcAddress((const GLubyte*)"glUniform1f")) == NULL) || r;
  r = ((glUniform1fv = (PFNGLUNIFORM1FVPROC)glewGetProcAddress((const GLubyte*)"glUniform1fv")) == NULL) || r;
  r = ((glUniform1i = (PFNGLUNIFORM1IPROC)glewGetProcAddress((const GLubyte*)"glUniform1i")) == NULL) || r;
  r = ((glUniform1iv = (PFNGLUNIFORM1IVPROC)glewGetProcAddress((const GLubyte*)"glUniform1iv")) == NULL) || r;
  r = ((glUniform2f = (PFNGLUNIFORM2FPROC)glewGetProcAddress((const GLubyte*)"glUniform2f")) == NULL) || r;
  r = ((glUniform2fv = (PFNGLUNIFORM2FVPROC)glewGetProcAddress((const GLubyte*)"glUniform2fv")) == NULL) || r;
  r = ((glUniform2i = (PFNGLUNIFORM2IPROC)glewGetProcAddress((const GLubyte*)"glUniform2i")) == NULL) || r;
  r = ((glUniform2iv = (PFNGLUNIFORM2IVPROC)glewGetProcAddress((const GLubyte*)"glUniform2iv")) == NULL) || r;
  r = ((glUniform3f = (PFNGLUNIFORM3FPROC)glewGetProcAddress((const GLubyte*)"glUniform3f")) == NULL) || r;
  r = ((glUniform3fv = (PFNGLUNIFORM3FVPROC)glewGetProcAddress((const GLubyte*)"glUniform3fv")) == NULL) || r;
  r = ((glUniform3i = (PFNGLUNIFORM3IPROC)glewGetProcAddress((const GLubyte*)"glUniform3i")) == NULL) || r;
  r = ((glUniform3iv = (PFNGLUNIFORM3IVPROC)glewGetProcAddress((const GLubyte*)"glUniform3iv")) == NULL) || r;
  r = ((glUniform4f = (PFNGLUNIFORM4FPROC)glewGetProcAddress((const GLubyte*)"glUniform4f")) == NULL) || r;
  r = ((glUniform4fv = (PFNGLUNIFORM4FVPROC)glewGetProcAddress((const GLubyte*)"glUniform4fv")) == NULL) || r;
  r = ((glUniform4i = (PFNGLUNIFORM4IPROC)glewGetProcAddress((const GLubyte*)"glUniform4i")) == NULL) || r;
  r = ((glUniform4iv = (PFNGLUNIFORM4IVPROC)glewGetProcAddress((const GLubyte*)"glUniform4iv")) == NULL) || r;
  r = ((glUniformMatrix2fv = (PFNGLUNIFORMMATRIX2FVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix2fv")) == NULL) || r;
  r = ((glUniformMatrix3fv = (PFNGLUNIFORMMATRIX3FVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix3fv")) == NULL) || r;
  r = ((glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix4fv")) == NULL) || r;
  r = ((glUseProgram = (PFNGLUSEPROGRAMPROC)glewGetProcAddress((const GLubyte*)"glUseProgram")) == NULL) || r;
  r = ((glValidateProgram = (PFNGLVALIDATEPROGRAMPROC)glewGetProcAddress((const GLubyte*)"glValidateProgram")) == NULL) || r;
  r = ((glVertexAttrib1d = (PFNGLVERTEXATTRIB1DPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1d")) == NULL) || r;
  r = ((glVertexAttrib1dv = (PFNGLVERTEXATTRIB1DVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1dv")) == NULL) || r;
  r = ((glVertexAttrib1f = (PFNGLVERTEXATTRIB1FPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1f")) == NULL) || r;
  r = ((glVertexAttrib1fv = (PFNGLVERTEXATTRIB1FVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1fv")) == NULL) || r;
  r = ((glVertexAttrib1s = (PFNGLVERTEXATTRIB1SPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1s")) == NULL) || r;
  r = ((glVertexAttrib1sv = (PFNGLVERTEXATTRIB1SVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1sv")) == NULL) || r;
  r = ((glVertexAttrib2d = (PFNGLVERTEXATTRIB2DPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2d")) == NULL) || r;
  r = ((glVertexAttrib2dv = (PFNGLVERTEXATTRIB2DVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2dv")) == NULL) || r;
  r = ((glVertexAttrib2f = (PFNGLVERTEXATTRIB2FPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2f")) == NULL) || r;
  r = ((glVertexAttrib2fv = (PFNGLVERTEXATTRIB2FVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2fv")) == NULL) || r;
  r = ((glVertexAttrib2s = (PFNGLVERTEXATTRIB2SPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2s")) == NULL) || r;
  r = ((glVertexAttrib2sv = (PFNGLVERTEXATTRIB2SVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2sv")) == NULL) || r;
  r = ((glVertexAttrib3d = (PFNGLVERTEXATTRIB3DPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3d")) == NULL) || r;
  r = ((glVertexAttrib3dv = (PFNGLVERTEXATTRIB3DVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3dv")) == NULL) || r;
  r = ((glVertexAttrib3f = (PFNGLVERTEXATTRIB3FPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3f")) == NULL) || r;
  r = ((glVertexAttrib3fv = (PFNGLVERTEXATTRIB3FVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3fv")) == NULL) || r;
  r = ((glVertexAttrib3s = (PFNGLVERTEXATTRIB3SPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3s")) == NULL) || r;
  r = ((glVertexAttrib3sv = (PFNGLVERTEXATTRIB3SVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3sv")) == NULL) || r;
  r = ((glVertexAttrib4Nbv = (PFNGLVERTEXATTRIB4NBVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4Nbv")) == NULL) || r;
  r = ((glVertexAttrib4Niv = (PFNGLVERTEXATTRIB4NIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4Niv")) == NULL) || r;
  r = ((glVertexAttrib4Nsv = (PFNGLVERTEXATTRIB4NSVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4Nsv")) == NULL) || r;
  r = ((glVertexAttrib4Nub = (PFNGLVERTEXATTRIB4NUBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4Nub")) == NULL) || r;
  r = ((glVertexAttrib4Nubv = (PFNGLVERTEXATTRIB4NUBVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4Nubv")) == NULL) || r;
  r = ((glVertexAttrib4Nuiv = (PFNGLVERTEXATTRIB4NUIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4Nuiv")) == NULL) || r;
  r = ((glVertexAttrib4Nusv = (PFNGLVERTEXATTRIB4NUSVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4Nusv")) == NULL) || r;
  r = ((glVertexAttrib4bv = (PFNGLVERTEXATTRIB4BVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4bv")) == NULL) || r;
  r = ((glVertexAttrib4d = (PFNGLVERTEXATTRIB4DPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4d")) == NULL) || r;
  r = ((glVertexAttrib4dv = (PFNGLVERTEXATTRIB4DVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4dv")) == NULL) || r;
  r = ((glVertexAttrib4f = (PFNGLVERTEXATTRIB4FPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4f")) == NULL) || r;
  r = ((glVertexAttrib4fv = (PFNGLVERTEXATTRIB4FVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4fv")) == NULL) || r;
  r = ((glVertexAttrib4iv = (PFNGLVERTEXATTRIB4IVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4iv")) == NULL) || r;
  r = ((glVertexAttrib4s = (PFNGLVERTEXATTRIB4SPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4s")) == NULL) || r;
  r = ((glVertexAttrib4sv = (PFNGLVERTEXATTRIB4SVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4sv")) == NULL) || r;
  r = ((glVertexAttrib4ubv = (PFNGLVERTEXATTRIB4UBVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4ubv")) == NULL) || r;
  r = ((glVertexAttrib4uiv = (PFNGLVERTEXATTRIB4UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4uiv")) == NULL) || r;
  r = ((glVertexAttrib4usv = (PFNGLVERTEXATTRIB4USVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4usv")) == NULL) || r;
  r = ((glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribPointer")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_2_0 */

#ifdef GL_VERSION_2_1

static GLboolean _glewInit_GL_VERSION_2_1 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glUniformMatrix2x3fv = (PFNGLUNIFORMMATRIX2X3FVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix2x3fv")) == NULL) || r;
  r = ((glUniformMatrix2x4fv = (PFNGLUNIFORMMATRIX2X4FVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix2x4fv")) == NULL) || r;
  r = ((glUniformMatrix3x2fv = (PFNGLUNIFORMMATRIX3X2FVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix3x2fv")) == NULL) || r;
  r = ((glUniformMatrix3x4fv = (PFNGLUNIFORMMATRIX3X4FVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix3x4fv")) == NULL) || r;
  r = ((glUniformMatrix4x2fv = (PFNGLUNIFORMMATRIX4X2FVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix4x2fv")) == NULL) || r;
  r = ((glUniformMatrix4x3fv = (PFNGLUNIFORMMATRIX4X3FVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix4x3fv")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_2_1 */

#ifdef GL_VERSION_3_0

static GLboolean _glewInit_GL_VERSION_3_0 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginConditionalRender = (PFNGLBEGINCONDITIONALRENDERPROC)glewGetProcAddress((const GLubyte*)"glBeginConditionalRender")) == NULL) || r;
  r = ((glBeginTransformFeedback = (PFNGLBEGINTRANSFORMFEEDBACKPROC)glewGetProcAddress((const GLubyte*)"glBeginTransformFeedback")) == NULL) || r;
  r = ((glBindFragDataLocation = (PFNGLBINDFRAGDATALOCATIONPROC)glewGetProcAddress((const GLubyte*)"glBindFragDataLocation")) == NULL) || r;
  r = ((glClampColor = (PFNGLCLAMPCOLORPROC)glewGetProcAddress((const GLubyte*)"glClampColor")) == NULL) || r;
  r = ((glClearBufferfi = (PFNGLCLEARBUFFERFIPROC)glewGetProcAddress((const GLubyte*)"glClearBufferfi")) == NULL) || r;
  r = ((glClearBufferfv = (PFNGLCLEARBUFFERFVPROC)glewGetProcAddress((const GLubyte*)"glClearBufferfv")) == NULL) || r;
  r = ((glClearBufferiv = (PFNGLCLEARBUFFERIVPROC)glewGetProcAddress((const GLubyte*)"glClearBufferiv")) == NULL) || r;
  r = ((glClearBufferuiv = (PFNGLCLEARBUFFERUIVPROC)glewGetProcAddress((const GLubyte*)"glClearBufferuiv")) == NULL) || r;
  r = ((glColorMaski = (PFNGLCOLORMASKIPROC)glewGetProcAddress((const GLubyte*)"glColorMaski")) == NULL) || r;
  r = ((glDisablei = (PFNGLDISABLEIPROC)glewGetProcAddress((const GLubyte*)"glDisablei")) == NULL) || r;
  r = ((glEnablei = (PFNGLENABLEIPROC)glewGetProcAddress((const GLubyte*)"glEnablei")) == NULL) || r;
  r = ((glEndConditionalRender = (PFNGLENDCONDITIONALRENDERPROC)glewGetProcAddress((const GLubyte*)"glEndConditionalRender")) == NULL) || r;
  r = ((glEndTransformFeedback = (PFNGLENDTRANSFORMFEEDBACKPROC)glewGetProcAddress((const GLubyte*)"glEndTransformFeedback")) == NULL) || r;
  r = ((glGetBooleani_v = (PFNGLGETBOOLEANI_VPROC)glewGetProcAddress((const GLubyte*)"glGetBooleani_v")) == NULL) || r;
  r = ((glGetFragDataLocation = (PFNGLGETFRAGDATALOCATIONPROC)glewGetProcAddress((const GLubyte*)"glGetFragDataLocation")) == NULL) || r;
  r = ((glGetStringi = (PFNGLGETSTRINGIPROC)glewGetProcAddress((const GLubyte*)"glGetStringi")) == NULL) || r;
  r = ((glGetTexParameterIiv = (PFNGLGETTEXPARAMETERIIVPROC)glewGetProcAddress((const GLubyte*)"glGetTexParameterIiv")) == NULL) || r;
  r = ((glGetTexParameterIuiv = (PFNGLGETTEXPARAMETERIUIVPROC)glewGetProcAddress((const GLubyte*)"glGetTexParameterIuiv")) == NULL) || r;
  r = ((glGetTransformFeedbackVarying = (PFNGLGETTRANSFORMFEEDBACKVARYINGPROC)glewGetProcAddress((const GLubyte*)"glGetTransformFeedbackVarying")) == NULL) || r;
  r = ((glGetUniformuiv = (PFNGLGETUNIFORMUIVPROC)glewGetProcAddress((const GLubyte*)"glGetUniformuiv")) == NULL) || r;
  r = ((glGetVertexAttribIiv = (PFNGLGETVERTEXATTRIBIIVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribIiv")) == NULL) || r;
  r = ((glGetVertexAttribIuiv = (PFNGLGETVERTEXATTRIBIUIVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribIuiv")) == NULL) || r;
  r = ((glIsEnabledi = (PFNGLISENABLEDIPROC)glewGetProcAddress((const GLubyte*)"glIsEnabledi")) == NULL) || r;
  r = ((glTexParameterIiv = (PFNGLTEXPARAMETERIIVPROC)glewGetProcAddress((const GLubyte*)"glTexParameterIiv")) == NULL) || r;
  r = ((glTexParameterIuiv = (PFNGLTEXPARAMETERIUIVPROC)glewGetProcAddress((const GLubyte*)"glTexParameterIuiv")) == NULL) || r;
  r = ((glTransformFeedbackVaryings = (PFNGLTRANSFORMFEEDBACKVARYINGSPROC)glewGetProcAddress((const GLubyte*)"glTransformFeedbackVaryings")) == NULL) || r;
  r = ((glUniform1ui = (PFNGLUNIFORM1UIPROC)glewGetProcAddress((const GLubyte*)"glUniform1ui")) == NULL) || r;
  r = ((glUniform1uiv = (PFNGLUNIFORM1UIVPROC)glewGetProcAddress((const GLubyte*)"glUniform1uiv")) == NULL) || r;
  r = ((glUniform2ui = (PFNGLUNIFORM2UIPROC)glewGetProcAddress((const GLubyte*)"glUniform2ui")) == NULL) || r;
  r = ((glUniform2uiv = (PFNGLUNIFORM2UIVPROC)glewGetProcAddress((const GLubyte*)"glUniform2uiv")) == NULL) || r;
  r = ((glUniform3ui = (PFNGLUNIFORM3UIPROC)glewGetProcAddress((const GLubyte*)"glUniform3ui")) == NULL) || r;
  r = ((glUniform3uiv = (PFNGLUNIFORM3UIVPROC)glewGetProcAddress((const GLubyte*)"glUniform3uiv")) == NULL) || r;
  r = ((glUniform4ui = (PFNGLUNIFORM4UIPROC)glewGetProcAddress((const GLubyte*)"glUniform4ui")) == NULL) || r;
  r = ((glUniform4uiv = (PFNGLUNIFORM4UIVPROC)glewGetProcAddress((const GLubyte*)"glUniform4uiv")) == NULL) || r;
  r = ((glVertexAttribI1i = (PFNGLVERTEXATTRIBI1IPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI1i")) == NULL) || r;
  r = ((glVertexAttribI1iv = (PFNGLVERTEXATTRIBI1IVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI1iv")) == NULL) || r;
  r = ((glVertexAttribI1ui = (PFNGLVERTEXATTRIBI1UIPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI1ui")) == NULL) || r;
  r = ((glVertexAttribI1uiv = (PFNGLVERTEXATTRIBI1UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI1uiv")) == NULL) || r;
  r = ((glVertexAttribI2i = (PFNGLVERTEXATTRIBI2IPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI2i")) == NULL) || r;
  r = ((glVertexAttribI2iv = (PFNGLVERTEXATTRIBI2IVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI2iv")) == NULL) || r;
  r = ((glVertexAttribI2ui = (PFNGLVERTEXATTRIBI2UIPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI2ui")) == NULL) || r;
  r = ((glVertexAttribI2uiv = (PFNGLVERTEXATTRIBI2UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI2uiv")) == NULL) || r;
  r = ((glVertexAttribI3i = (PFNGLVERTEXATTRIBI3IPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI3i")) == NULL) || r;
  r = ((glVertexAttribI3iv = (PFNGLVERTEXATTRIBI3IVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI3iv")) == NULL) || r;
  r = ((glVertexAttribI3ui = (PFNGLVERTEXATTRIBI3UIPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI3ui")) == NULL) || r;
  r = ((glVertexAttribI3uiv = (PFNGLVERTEXATTRIBI3UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI3uiv")) == NULL) || r;
  r = ((glVertexAttribI4bv = (PFNGLVERTEXATTRIBI4BVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4bv")) == NULL) || r;
  r = ((glVertexAttribI4i = (PFNGLVERTEXATTRIBI4IPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4i")) == NULL) || r;
  r = ((glVertexAttribI4iv = (PFNGLVERTEXATTRIBI4IVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4iv")) == NULL) || r;
  r = ((glVertexAttribI4sv = (PFNGLVERTEXATTRIBI4SVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4sv")) == NULL) || r;
  r = ((glVertexAttribI4ubv = (PFNGLVERTEXATTRIBI4UBVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4ubv")) == NULL) || r;
  r = ((glVertexAttribI4ui = (PFNGLVERTEXATTRIBI4UIPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4ui")) == NULL) || r;
  r = ((glVertexAttribI4uiv = (PFNGLVERTEXATTRIBI4UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4uiv")) == NULL) || r;
  r = ((glVertexAttribI4usv = (PFNGLVERTEXATTRIBI4USVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4usv")) == NULL) || r;
  r = ((glVertexAttribIPointer = (PFNGLVERTEXATTRIBIPOINTERPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribIPointer")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_3_0 */

#ifdef GL_VERSION_3_1

static GLboolean _glewInit_GL_VERSION_3_1 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawArraysInstanced = (PFNGLDRAWARRAYSINSTANCEDPROC)glewGetProcAddress((const GLubyte*)"glDrawArraysInstanced")) == NULL) || r;
  r = ((glDrawElementsInstanced = (PFNGLDRAWELEMENTSINSTANCEDPROC)glewGetProcAddress((const GLubyte*)"glDrawElementsInstanced")) == NULL) || r;
  r = ((glPrimitiveRestartIndex = (PFNGLPRIMITIVERESTARTINDEXPROC)glewGetProcAddress((const GLubyte*)"glPrimitiveRestartIndex")) == NULL) || r;
  r = ((glTexBuffer = (PFNGLTEXBUFFERPROC)glewGetProcAddress((const GLubyte*)"glTexBuffer")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_3_1 */

#ifdef GL_VERSION_3_2

static GLboolean _glewInit_GL_VERSION_3_2 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFramebufferTexture = (PFNGLFRAMEBUFFERTEXTUREPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTexture")) == NULL) || r;
  r = ((glGetBufferParameteri64v = (PFNGLGETBUFFERPARAMETERI64VPROC)glewGetProcAddress((const GLubyte*)"glGetBufferParameteri64v")) == NULL) || r;
  r = ((glGetInteger64i_v = (PFNGLGETINTEGER64I_VPROC)glewGetProcAddress((const GLubyte*)"glGetInteger64i_v")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_3_2 */

#ifdef GL_VERSION_3_3

static GLboolean _glewInit_GL_VERSION_3_3 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glVertexAttribDivisor = (PFNGLVERTEXATTRIBDIVISORPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribDivisor")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_3_3 */

#ifdef GL_VERSION_4_0

static GLboolean _glewInit_GL_VERSION_4_0 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendEquationSeparatei = (PFNGLBLENDEQUATIONSEPARATEIPROC)glewGetProcAddress((const GLubyte*)"glBlendEquationSeparatei")) == NULL) || r;
  r = ((glBlendEquationi = (PFNGLBLENDEQUATIONIPROC)glewGetProcAddress((const GLubyte*)"glBlendEquationi")) == NULL) || r;
  r = ((glBlendFuncSeparatei = (PFNGLBLENDFUNCSEPARATEIPROC)glewGetProcAddress((const GLubyte*)"glBlendFuncSeparatei")) == NULL) || r;
  r = ((glBlendFunci = (PFNGLBLENDFUNCIPROC)glewGetProcAddress((const GLubyte*)"glBlendFunci")) == NULL) || r;
  r = ((glMinSampleShading = (PFNGLMINSAMPLESHADINGPROC)glewGetProcAddress((const GLubyte*)"glMinSampleShading")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_4_0 */

#ifdef GL_VERSION_4_5

static GLboolean _glewInit_GL_VERSION_4_5 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetGraphicsResetStatus = (PFNGLGETGRAPHICSRESETSTATUSPROC)glewGetProcAddress((const GLubyte*)"glGetGraphicsResetStatus")) == NULL) || r;
  r = ((glGetnCompressedTexImage = (PFNGLGETNCOMPRESSEDTEXIMAGEPROC)glewGetProcAddress((const GLubyte*)"glGetnCompressedTexImage")) == NULL) || r;
  r = ((glGetnTexImage = (PFNGLGETNTEXIMAGEPROC)glewGetProcAddress((const GLubyte*)"glGetnTexImage")) == NULL) || r;
  r = ((glGetnUniformdv = (PFNGLGETNUNIFORMDVPROC)glewGetProcAddress((const GLubyte*)"glGetnUniformdv")) == NULL) || r;

  return r;
}

#endif /* GL_VERSION_4_5 */

#ifdef GL_3DFX_tbuffer

static GLboolean _glewInit_GL_3DFX_tbuffer (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTbufferMask3DFX = (PFNGLTBUFFERMASK3DFXPROC)glewGetProcAddress((const GLubyte*)"glTbufferMask3DFX")) == NULL) || r;

  return r;
}

#endif /* GL_3DFX_tbuffer */

#ifdef GL_AMD_debug_output

static GLboolean _glewInit_GL_AMD_debug_output (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDebugMessageCallbackAMD = (PFNGLDEBUGMESSAGECALLBACKAMDPROC)glewGetProcAddress((const GLubyte*)"glDebugMessageCallbackAMD")) == NULL) || r;
  r = ((glDebugMessageEnableAMD = (PFNGLDEBUGMESSAGEENABLEAMDPROC)glewGetProcAddress((const GLubyte*)"glDebugMessageEnableAMD")) == NULL) || r;
  r = ((glDebugMessageInsertAMD = (PFNGLDEBUGMESSAGEINSERTAMDPROC)glewGetProcAddress((const GLubyte*)"glDebugMessageInsertAMD")) == NULL) || r;
  r = ((glGetDebugMessageLogAMD = (PFNGLGETDEBUGMESSAGELOGAMDPROC)glewGetProcAddress((const GLubyte*)"glGetDebugMessageLogAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_debug_output */

#ifdef GL_AMD_draw_buffers_blend

static GLboolean _glewInit_GL_AMD_draw_buffers_blend (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendEquationIndexedAMD = (PFNGLBLENDEQUATIONINDEXEDAMDPROC)glewGetProcAddress((const GLubyte*)"glBlendEquationIndexedAMD")) == NULL) || r;
  r = ((glBlendEquationSeparateIndexedAMD = (PFNGLBLENDEQUATIONSEPARATEINDEXEDAMDPROC)glewGetProcAddress((const GLubyte*)"glBlendEquationSeparateIndexedAMD")) == NULL) || r;
  r = ((glBlendFuncIndexedAMD = (PFNGLBLENDFUNCINDEXEDAMDPROC)glewGetProcAddress((const GLubyte*)"glBlendFuncIndexedAMD")) == NULL) || r;
  r = ((glBlendFuncSeparateIndexedAMD = (PFNGLBLENDFUNCSEPARATEINDEXEDAMDPROC)glewGetProcAddress((const GLubyte*)"glBlendFuncSeparateIndexedAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_draw_buffers_blend */

#ifdef GL_AMD_interleaved_elements

static GLboolean _glewInit_GL_AMD_interleaved_elements (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glVertexAttribParameteriAMD = (PFNGLVERTEXATTRIBPARAMETERIAMDPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribParameteriAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_interleaved_elements */

#ifdef GL_AMD_multi_draw_indirect

static GLboolean _glewInit_GL_AMD_multi_draw_indirect (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMultiDrawArraysIndirectAMD = (PFNGLMULTIDRAWARRAYSINDIRECTAMDPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawArraysIndirectAMD")) == NULL) || r;
  r = ((glMultiDrawElementsIndirectAMD = (PFNGLMULTIDRAWELEMENTSINDIRECTAMDPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawElementsIndirectAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_multi_draw_indirect */

#ifdef GL_AMD_name_gen_delete

static GLboolean _glewInit_GL_AMD_name_gen_delete (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDeleteNamesAMD = (PFNGLDELETENAMESAMDPROC)glewGetProcAddress((const GLubyte*)"glDeleteNamesAMD")) == NULL) || r;
  r = ((glGenNamesAMD = (PFNGLGENNAMESAMDPROC)glewGetProcAddress((const GLubyte*)"glGenNamesAMD")) == NULL) || r;
  r = ((glIsNameAMD = (PFNGLISNAMEAMDPROC)glewGetProcAddress((const GLubyte*)"glIsNameAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_name_gen_delete */

#ifdef GL_AMD_occlusion_query_event

static GLboolean _glewInit_GL_AMD_occlusion_query_event (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glQueryObjectParameteruiAMD = (PFNGLQUERYOBJECTPARAMETERUIAMDPROC)glewGetProcAddress((const GLubyte*)"glQueryObjectParameteruiAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_occlusion_query_event */

#ifdef GL_AMD_performance_monitor

static GLboolean _glewInit_GL_AMD_performance_monitor (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginPerfMonitorAMD = (PFNGLBEGINPERFMONITORAMDPROC)glewGetProcAddress((const GLubyte*)"glBeginPerfMonitorAMD")) == NULL) || r;
  r = ((glDeletePerfMonitorsAMD = (PFNGLDELETEPERFMONITORSAMDPROC)glewGetProcAddress((const GLubyte*)"glDeletePerfMonitorsAMD")) == NULL) || r;
  r = ((glEndPerfMonitorAMD = (PFNGLENDPERFMONITORAMDPROC)glewGetProcAddress((const GLubyte*)"glEndPerfMonitorAMD")) == NULL) || r;
  r = ((glGenPerfMonitorsAMD = (PFNGLGENPERFMONITORSAMDPROC)glewGetProcAddress((const GLubyte*)"glGenPerfMonitorsAMD")) == NULL) || r;
  r = ((glGetPerfMonitorCounterDataAMD = (PFNGLGETPERFMONITORCOUNTERDATAAMDPROC)glewGetProcAddress((const GLubyte*)"glGetPerfMonitorCounterDataAMD")) == NULL) || r;
  r = ((glGetPerfMonitorCounterInfoAMD = (PFNGLGETPERFMONITORCOUNTERINFOAMDPROC)glewGetProcAddress((const GLubyte*)"glGetPerfMonitorCounterInfoAMD")) == NULL) || r;
  r = ((glGetPerfMonitorCounterStringAMD = (PFNGLGETPERFMONITORCOUNTERSTRINGAMDPROC)glewGetProcAddress((const GLubyte*)"glGetPerfMonitorCounterStringAMD")) == NULL) || r;
  r = ((glGetPerfMonitorCountersAMD = (PFNGLGETPERFMONITORCOUNTERSAMDPROC)glewGetProcAddress((const GLubyte*)"glGetPerfMonitorCountersAMD")) == NULL) || r;
  r = ((glGetPerfMonitorGroupStringAMD = (PFNGLGETPERFMONITORGROUPSTRINGAMDPROC)glewGetProcAddress((const GLubyte*)"glGetPerfMonitorGroupStringAMD")) == NULL) || r;
  r = ((glGetPerfMonitorGroupsAMD = (PFNGLGETPERFMONITORGROUPSAMDPROC)glewGetProcAddress((const GLubyte*)"glGetPerfMonitorGroupsAMD")) == NULL) || r;
  r = ((glSelectPerfMonitorCountersAMD = (PFNGLSELECTPERFMONITORCOUNTERSAMDPROC)glewGetProcAddress((const GLubyte*)"glSelectPerfMonitorCountersAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_performance_monitor */

#ifdef GL_AMD_sample_positions

static GLboolean _glewInit_GL_AMD_sample_positions (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glSetMultisamplefvAMD = (PFNGLSETMULTISAMPLEFVAMDPROC)glewGetProcAddress((const GLubyte*)"glSetMultisamplefvAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_sample_positions */

#ifdef GL_AMD_sparse_texture

static GLboolean _glewInit_GL_AMD_sparse_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexStorageSparseAMD = (PFNGLTEXSTORAGESPARSEAMDPROC)glewGetProcAddress((const GLubyte*)"glTexStorageSparseAMD")) == NULL) || r;
  r = ((glTextureStorageSparseAMD = (PFNGLTEXTURESTORAGESPARSEAMDPROC)glewGetProcAddress((const GLubyte*)"glTextureStorageSparseAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_sparse_texture */

#ifdef GL_AMD_stencil_operation_extended

static GLboolean _glewInit_GL_AMD_stencil_operation_extended (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glStencilOpValueAMD = (PFNGLSTENCILOPVALUEAMDPROC)glewGetProcAddress((const GLubyte*)"glStencilOpValueAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_stencil_operation_extended */

#ifdef GL_AMD_vertex_shader_tessellator

static GLboolean _glewInit_GL_AMD_vertex_shader_tessellator (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTessellationFactorAMD = (PFNGLTESSELLATIONFACTORAMDPROC)glewGetProcAddress((const GLubyte*)"glTessellationFactorAMD")) == NULL) || r;
  r = ((glTessellationModeAMD = (PFNGLTESSELLATIONMODEAMDPROC)glewGetProcAddress((const GLubyte*)"glTessellationModeAMD")) == NULL) || r;

  return r;
}

#endif /* GL_AMD_vertex_shader_tessellator */

#ifdef GL_ANGLE_framebuffer_blit

static GLboolean _glewInit_GL_ANGLE_framebuffer_blit (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlitFramebufferANGLE = (PFNGLBLITFRAMEBUFFERANGLEPROC)glewGetProcAddress((const GLubyte*)"glBlitFramebufferANGLE")) == NULL) || r;

  return r;
}

#endif /* GL_ANGLE_framebuffer_blit */

#ifdef GL_ANGLE_framebuffer_multisample

static GLboolean _glewInit_GL_ANGLE_framebuffer_multisample (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glRenderbufferStorageMultisampleANGLE = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLEANGLEPROC)glewGetProcAddress((const GLubyte*)"glRenderbufferStorageMultisampleANGLE")) == NULL) || r;

  return r;
}

#endif /* GL_ANGLE_framebuffer_multisample */

#ifdef GL_ANGLE_instanced_arrays

static GLboolean _glewInit_GL_ANGLE_instanced_arrays (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawArraysInstancedANGLE = (PFNGLDRAWARRAYSINSTANCEDANGLEPROC)glewGetProcAddress((const GLubyte*)"glDrawArraysInstancedANGLE")) == NULL) || r;
  r = ((glDrawElementsInstancedANGLE = (PFNGLDRAWELEMENTSINSTANCEDANGLEPROC)glewGetProcAddress((const GLubyte*)"glDrawElementsInstancedANGLE")) == NULL) || r;
  r = ((glVertexAttribDivisorANGLE = (PFNGLVERTEXATTRIBDIVISORANGLEPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribDivisorANGLE")) == NULL) || r;

  return r;
}

#endif /* GL_ANGLE_instanced_arrays */

#ifdef GL_ANGLE_timer_query

static GLboolean _glewInit_GL_ANGLE_timer_query (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginQueryANGLE = (PFNGLBEGINQUERYANGLEPROC)glewGetProcAddress((const GLubyte*)"glBeginQueryANGLE")) == NULL) || r;
  r = ((glDeleteQueriesANGLE = (PFNGLDELETEQUERIESANGLEPROC)glewGetProcAddress((const GLubyte*)"glDeleteQueriesANGLE")) == NULL) || r;
  r = ((glEndQueryANGLE = (PFNGLENDQUERYANGLEPROC)glewGetProcAddress((const GLubyte*)"glEndQueryANGLE")) == NULL) || r;
  r = ((glGenQueriesANGLE = (PFNGLGENQUERIESANGLEPROC)glewGetProcAddress((const GLubyte*)"glGenQueriesANGLE")) == NULL) || r;
  r = ((glGetQueryObjecti64vANGLE = (PFNGLGETQUERYOBJECTI64VANGLEPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjecti64vANGLE")) == NULL) || r;
  r = ((glGetQueryObjectivANGLE = (PFNGLGETQUERYOBJECTIVANGLEPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjectivANGLE")) == NULL) || r;
  r = ((glGetQueryObjectui64vANGLE = (PFNGLGETQUERYOBJECTUI64VANGLEPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjectui64vANGLE")) == NULL) || r;
  r = ((glGetQueryObjectuivANGLE = (PFNGLGETQUERYOBJECTUIVANGLEPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjectuivANGLE")) == NULL) || r;
  r = ((glGetQueryivANGLE = (PFNGLGETQUERYIVANGLEPROC)glewGetProcAddress((const GLubyte*)"glGetQueryivANGLE")) == NULL) || r;
  r = ((glIsQueryANGLE = (PFNGLISQUERYANGLEPROC)glewGetProcAddress((const GLubyte*)"glIsQueryANGLE")) == NULL) || r;
  r = ((glQueryCounterANGLE = (PFNGLQUERYCOUNTERANGLEPROC)glewGetProcAddress((const GLubyte*)"glQueryCounterANGLE")) == NULL) || r;

  return r;
}

#endif /* GL_ANGLE_timer_query */

#ifdef GL_ANGLE_translated_shader_source

static GLboolean _glewInit_GL_ANGLE_translated_shader_source (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetTranslatedShaderSourceANGLE = (PFNGLGETTRANSLATEDSHADERSOURCEANGLEPROC)glewGetProcAddress((const GLubyte*)"glGetTranslatedShaderSourceANGLE")) == NULL) || r;

  return r;
}

#endif /* GL_ANGLE_translated_shader_source */

#ifdef GL_APPLE_element_array

static GLboolean _glewInit_GL_APPLE_element_array (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawElementArrayAPPLE = (PFNGLDRAWELEMENTARRAYAPPLEPROC)glewGetProcAddress((const GLubyte*)"glDrawElementArrayAPPLE")) == NULL) || r;
  r = ((glDrawRangeElementArrayAPPLE = (PFNGLDRAWRANGEELEMENTARRAYAPPLEPROC)glewGetProcAddress((const GLubyte*)"glDrawRangeElementArrayAPPLE")) == NULL) || r;
  r = ((glElementPointerAPPLE = (PFNGLELEMENTPOINTERAPPLEPROC)glewGetProcAddress((const GLubyte*)"glElementPointerAPPLE")) == NULL) || r;
  r = ((glMultiDrawElementArrayAPPLE = (PFNGLMULTIDRAWELEMENTARRAYAPPLEPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawElementArrayAPPLE")) == NULL) || r;
  r = ((glMultiDrawRangeElementArrayAPPLE = (PFNGLMULTIDRAWRANGEELEMENTARRAYAPPLEPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawRangeElementArrayAPPLE")) == NULL) || r;

  return r;
}

#endif /* GL_APPLE_element_array */

#ifdef GL_APPLE_fence

static GLboolean _glewInit_GL_APPLE_fence (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDeleteFencesAPPLE = (PFNGLDELETEFENCESAPPLEPROC)glewGetProcAddress((const GLubyte*)"glDeleteFencesAPPLE")) == NULL) || r;
  r = ((glFinishFenceAPPLE = (PFNGLFINISHFENCEAPPLEPROC)glewGetProcAddress((const GLubyte*)"glFinishFenceAPPLE")) == NULL) || r;
  r = ((glFinishObjectAPPLE = (PFNGLFINISHOBJECTAPPLEPROC)glewGetProcAddress((const GLubyte*)"glFinishObjectAPPLE")) == NULL) || r;
  r = ((glGenFencesAPPLE = (PFNGLGENFENCESAPPLEPROC)glewGetProcAddress((const GLubyte*)"glGenFencesAPPLE")) == NULL) || r;
  r = ((glIsFenceAPPLE = (PFNGLISFENCEAPPLEPROC)glewGetProcAddress((const GLubyte*)"glIsFenceAPPLE")) == NULL) || r;
  r = ((glSetFenceAPPLE = (PFNGLSETFENCEAPPLEPROC)glewGetProcAddress((const GLubyte*)"glSetFenceAPPLE")) == NULL) || r;
  r = ((glTestFenceAPPLE = (PFNGLTESTFENCEAPPLEPROC)glewGetProcAddress((const GLubyte*)"glTestFenceAPPLE")) == NULL) || r;
  r = ((glTestObjectAPPLE = (PFNGLTESTOBJECTAPPLEPROC)glewGetProcAddress((const GLubyte*)"glTestObjectAPPLE")) == NULL) || r;

  return r;
}

#endif /* GL_APPLE_fence */

#ifdef GL_APPLE_flush_buffer_range

static GLboolean _glewInit_GL_APPLE_flush_buffer_range (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBufferParameteriAPPLE = (PFNGLBUFFERPARAMETERIAPPLEPROC)glewGetProcAddress((const GLubyte*)"glBufferParameteriAPPLE")) == NULL) || r;
  r = ((glFlushMappedBufferRangeAPPLE = (PFNGLFLUSHMAPPEDBUFFERRANGEAPPLEPROC)glewGetProcAddress((const GLubyte*)"glFlushMappedBufferRangeAPPLE")) == NULL) || r;

  return r;
}

#endif /* GL_APPLE_flush_buffer_range */

#ifdef GL_APPLE_object_purgeable

static GLboolean _glewInit_GL_APPLE_object_purgeable (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetObjectParameterivAPPLE = (PFNGLGETOBJECTPARAMETERIVAPPLEPROC)glewGetProcAddress((const GLubyte*)"glGetObjectParameterivAPPLE")) == NULL) || r;
  r = ((glObjectPurgeableAPPLE = (PFNGLOBJECTPURGEABLEAPPLEPROC)glewGetProcAddress((const GLubyte*)"glObjectPurgeableAPPLE")) == NULL) || r;
  r = ((glObjectUnpurgeableAPPLE = (PFNGLOBJECTUNPURGEABLEAPPLEPROC)glewGetProcAddress((const GLubyte*)"glObjectUnpurgeableAPPLE")) == NULL) || r;

  return r;
}

#endif /* GL_APPLE_object_purgeable */

#ifdef GL_APPLE_texture_range

static GLboolean _glewInit_GL_APPLE_texture_range (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetTexParameterPointervAPPLE = (PFNGLGETTEXPARAMETERPOINTERVAPPLEPROC)glewGetProcAddress((const GLubyte*)"glGetTexParameterPointervAPPLE")) == NULL) || r;
  r = ((glTextureRangeAPPLE = (PFNGLTEXTURERANGEAPPLEPROC)glewGetProcAddress((const GLubyte*)"glTextureRangeAPPLE")) == NULL) || r;

  return r;
}

#endif /* GL_APPLE_texture_range */

#ifdef GL_APPLE_vertex_array_object

static GLboolean _glewInit_GL_APPLE_vertex_array_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindVertexArrayAPPLE = (PFNGLBINDVERTEXARRAYAPPLEPROC)glewGetProcAddress((const GLubyte*)"glBindVertexArrayAPPLE")) == NULL) || r;
  r = ((glDeleteVertexArraysAPPLE = (PFNGLDELETEVERTEXARRAYSAPPLEPROC)glewGetProcAddress((const GLubyte*)"glDeleteVertexArraysAPPLE")) == NULL) || r;
  r = ((glGenVertexArraysAPPLE = (PFNGLGENVERTEXARRAYSAPPLEPROC)glewGetProcAddress((const GLubyte*)"glGenVertexArraysAPPLE")) == NULL) || r;
  r = ((glIsVertexArrayAPPLE = (PFNGLISVERTEXARRAYAPPLEPROC)glewGetProcAddress((const GLubyte*)"glIsVertexArrayAPPLE")) == NULL) || r;

  return r;
}

#endif /* GL_APPLE_vertex_array_object */

#ifdef GL_APPLE_vertex_array_range

static GLboolean _glewInit_GL_APPLE_vertex_array_range (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFlushVertexArrayRangeAPPLE = (PFNGLFLUSHVERTEXARRAYRANGEAPPLEPROC)glewGetProcAddress((const GLubyte*)"glFlushVertexArrayRangeAPPLE")) == NULL) || r;
  r = ((glVertexArrayParameteriAPPLE = (PFNGLVERTEXARRAYPARAMETERIAPPLEPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayParameteriAPPLE")) == NULL) || r;
  r = ((glVertexArrayRangeAPPLE = (PFNGLVERTEXARRAYRANGEAPPLEPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayRangeAPPLE")) == NULL) || r;

  return r;
}

#endif /* GL_APPLE_vertex_array_range */

#ifdef GL_APPLE_vertex_program_evaluators

static GLboolean _glewInit_GL_APPLE_vertex_program_evaluators (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDisableVertexAttribAPPLE = (PFNGLDISABLEVERTEXATTRIBAPPLEPROC)glewGetProcAddress((const GLubyte*)"glDisableVertexAttribAPPLE")) == NULL) || r;
  r = ((glEnableVertexAttribAPPLE = (PFNGLENABLEVERTEXATTRIBAPPLEPROC)glewGetProcAddress((const GLubyte*)"glEnableVertexAttribAPPLE")) == NULL) || r;
  r = ((glIsVertexAttribEnabledAPPLE = (PFNGLISVERTEXATTRIBENABLEDAPPLEPROC)glewGetProcAddress((const GLubyte*)"glIsVertexAttribEnabledAPPLE")) == NULL) || r;
  r = ((glMapVertexAttrib1dAPPLE = (PFNGLMAPVERTEXATTRIB1DAPPLEPROC)glewGetProcAddress((const GLubyte*)"glMapVertexAttrib1dAPPLE")) == NULL) || r;
  r = ((glMapVertexAttrib1fAPPLE = (PFNGLMAPVERTEXATTRIB1FAPPLEPROC)glewGetProcAddress((const GLubyte*)"glMapVertexAttrib1fAPPLE")) == NULL) || r;
  r = ((glMapVertexAttrib2dAPPLE = (PFNGLMAPVERTEXATTRIB2DAPPLEPROC)glewGetProcAddress((const GLubyte*)"glMapVertexAttrib2dAPPLE")) == NULL) || r;
  r = ((glMapVertexAttrib2fAPPLE = (PFNGLMAPVERTEXATTRIB2FAPPLEPROC)glewGetProcAddress((const GLubyte*)"glMapVertexAttrib2fAPPLE")) == NULL) || r;

  return r;
}

#endif /* GL_APPLE_vertex_program_evaluators */

#ifdef GL_ARB_ES2_compatibility

static GLboolean _glewInit_GL_ARB_ES2_compatibility (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClearDepthf = (PFNGLCLEARDEPTHFPROC)glewGetProcAddress((const GLubyte*)"glClearDepthf")) == NULL) || r;
  r = ((glDepthRangef = (PFNGLDEPTHRANGEFPROC)glewGetProcAddress((const GLubyte*)"glDepthRangef")) == NULL) || r;
  r = ((glGetShaderPrecisionFormat = (PFNGLGETSHADERPRECISIONFORMATPROC)glewGetProcAddress((const GLubyte*)"glGetShaderPrecisionFormat")) == NULL) || r;
  r = ((glReleaseShaderCompiler = (PFNGLRELEASESHADERCOMPILERPROC)glewGetProcAddress((const GLubyte*)"glReleaseShaderCompiler")) == NULL) || r;
  r = ((glShaderBinary = (PFNGLSHADERBINARYPROC)glewGetProcAddress((const GLubyte*)"glShaderBinary")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_ES2_compatibility */

#ifdef GL_ARB_ES3_1_compatibility

static GLboolean _glewInit_GL_ARB_ES3_1_compatibility (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMemoryBarrierByRegion = (PFNGLMEMORYBARRIERBYREGIONPROC)glewGetProcAddress((const GLubyte*)"glMemoryBarrierByRegion")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_ES3_1_compatibility */

#ifdef GL_ARB_ES3_2_compatibility

static GLboolean _glewInit_GL_ARB_ES3_2_compatibility (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glPrimitiveBoundingBoxARB = (PFNGLPRIMITIVEBOUNDINGBOXARBPROC)glewGetProcAddress((const GLubyte*)"glPrimitiveBoundingBoxARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_ES3_2_compatibility */

#ifdef GL_ARB_base_instance

static GLboolean _glewInit_GL_ARB_base_instance (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawArraysInstancedBaseInstance = (PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC)glewGetProcAddress((const GLubyte*)"glDrawArraysInstancedBaseInstance")) == NULL) || r;
  r = ((glDrawElementsInstancedBaseInstance = (PFNGLDRAWELEMENTSINSTANCEDBASEINSTANCEPROC)glewGetProcAddress((const GLubyte*)"glDrawElementsInstancedBaseInstance")) == NULL) || r;
  r = ((glDrawElementsInstancedBaseVertexBaseInstance = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC)glewGetProcAddress((const GLubyte*)"glDrawElementsInstancedBaseVertexBaseInstance")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_base_instance */

#ifdef GL_ARB_bindless_texture

static GLboolean _glewInit_GL_ARB_bindless_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetImageHandleARB = (PFNGLGETIMAGEHANDLEARBPROC)glewGetProcAddress((const GLubyte*)"glGetImageHandleARB")) == NULL) || r;
  r = ((glGetTextureHandleARB = (PFNGLGETTEXTUREHANDLEARBPROC)glewGetProcAddress((const GLubyte*)"glGetTextureHandleARB")) == NULL) || r;
  r = ((glGetTextureSamplerHandleARB = (PFNGLGETTEXTURESAMPLERHANDLEARBPROC)glewGetProcAddress((const GLubyte*)"glGetTextureSamplerHandleARB")) == NULL) || r;
  r = ((glGetVertexAttribLui64vARB = (PFNGLGETVERTEXATTRIBLUI64VARBPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribLui64vARB")) == NULL) || r;
  r = ((glIsImageHandleResidentARB = (PFNGLISIMAGEHANDLERESIDENTARBPROC)glewGetProcAddress((const GLubyte*)"glIsImageHandleResidentARB")) == NULL) || r;
  r = ((glIsTextureHandleResidentARB = (PFNGLISTEXTUREHANDLERESIDENTARBPROC)glewGetProcAddress((const GLubyte*)"glIsTextureHandleResidentARB")) == NULL) || r;
  r = ((glMakeImageHandleNonResidentARB = (PFNGLMAKEIMAGEHANDLENONRESIDENTARBPROC)glewGetProcAddress((const GLubyte*)"glMakeImageHandleNonResidentARB")) == NULL) || r;
  r = ((glMakeImageHandleResidentARB = (PFNGLMAKEIMAGEHANDLERESIDENTARBPROC)glewGetProcAddress((const GLubyte*)"glMakeImageHandleResidentARB")) == NULL) || r;
  r = ((glMakeTextureHandleNonResidentARB = (PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC)glewGetProcAddress((const GLubyte*)"glMakeTextureHandleNonResidentARB")) == NULL) || r;
  r = ((glMakeTextureHandleResidentARB = (PFNGLMAKETEXTUREHANDLERESIDENTARBPROC)glewGetProcAddress((const GLubyte*)"glMakeTextureHandleResidentARB")) == NULL) || r;
  r = ((glProgramUniformHandleui64ARB = (PFNGLPROGRAMUNIFORMHANDLEUI64ARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformHandleui64ARB")) == NULL) || r;
  r = ((glProgramUniformHandleui64vARB = (PFNGLPROGRAMUNIFORMHANDLEUI64VARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformHandleui64vARB")) == NULL) || r;
  r = ((glUniformHandleui64ARB = (PFNGLUNIFORMHANDLEUI64ARBPROC)glewGetProcAddress((const GLubyte*)"glUniformHandleui64ARB")) == NULL) || r;
  r = ((glUniformHandleui64vARB = (PFNGLUNIFORMHANDLEUI64VARBPROC)glewGetProcAddress((const GLubyte*)"glUniformHandleui64vARB")) == NULL) || r;
  r = ((glVertexAttribL1ui64ARB = (PFNGLVERTEXATTRIBL1UI64ARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL1ui64ARB")) == NULL) || r;
  r = ((glVertexAttribL1ui64vARB = (PFNGLVERTEXATTRIBL1UI64VARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL1ui64vARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_bindless_texture */

#ifdef GL_ARB_blend_func_extended

static GLboolean _glewInit_GL_ARB_blend_func_extended (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindFragDataLocationIndexed = (PFNGLBINDFRAGDATALOCATIONINDEXEDPROC)glewGetProcAddress((const GLubyte*)"glBindFragDataLocationIndexed")) == NULL) || r;
  r = ((glGetFragDataIndex = (PFNGLGETFRAGDATAINDEXPROC)glewGetProcAddress((const GLubyte*)"glGetFragDataIndex")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_blend_func_extended */

#ifdef GL_ARB_buffer_storage

static GLboolean _glewInit_GL_ARB_buffer_storage (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBufferStorage = (PFNGLBUFFERSTORAGEPROC)glewGetProcAddress((const GLubyte*)"glBufferStorage")) == NULL) || r;
  r = ((glNamedBufferStorageEXT = (PFNGLNAMEDBUFFERSTORAGEEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedBufferStorageEXT")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_buffer_storage */

#ifdef GL_ARB_cl_event

static GLboolean _glewInit_GL_ARB_cl_event (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCreateSyncFromCLeventARB = (PFNGLCREATESYNCFROMCLEVENTARBPROC)glewGetProcAddress((const GLubyte*)"glCreateSyncFromCLeventARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_cl_event */

#ifdef GL_ARB_clear_buffer_object

static GLboolean _glewInit_GL_ARB_clear_buffer_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClearBufferData = (PFNGLCLEARBUFFERDATAPROC)glewGetProcAddress((const GLubyte*)"glClearBufferData")) == NULL) || r;
  r = ((glClearBufferSubData = (PFNGLCLEARBUFFERSUBDATAPROC)glewGetProcAddress((const GLubyte*)"glClearBufferSubData")) == NULL) || r;
  r = ((glClearNamedBufferDataEXT = (PFNGLCLEARNAMEDBUFFERDATAEXTPROC)glewGetProcAddress((const GLubyte*)"glClearNamedBufferDataEXT")) == NULL) || r;
  r = ((glClearNamedBufferSubDataEXT = (PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC)glewGetProcAddress((const GLubyte*)"glClearNamedBufferSubDataEXT")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_clear_buffer_object */

#ifdef GL_ARB_clear_texture

static GLboolean _glewInit_GL_ARB_clear_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClearTexImage = (PFNGLCLEARTEXIMAGEPROC)glewGetProcAddress((const GLubyte*)"glClearTexImage")) == NULL) || r;
  r = ((glClearTexSubImage = (PFNGLCLEARTEXSUBIMAGEPROC)glewGetProcAddress((const GLubyte*)"glClearTexSubImage")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_clear_texture */

#ifdef GL_ARB_clip_control

static GLboolean _glewInit_GL_ARB_clip_control (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClipControl = (PFNGLCLIPCONTROLPROC)glewGetProcAddress((const GLubyte*)"glClipControl")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_clip_control */

#ifdef GL_ARB_color_buffer_float

static GLboolean _glewInit_GL_ARB_color_buffer_float (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClampColorARB = (PFNGLCLAMPCOLORARBPROC)glewGetProcAddress((const GLubyte*)"glClampColorARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_color_buffer_float */

#ifdef GL_ARB_compute_shader

static GLboolean _glewInit_GL_ARB_compute_shader (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)glewGetProcAddress((const GLubyte*)"glDispatchCompute")) == NULL) || r;
  r = ((glDispatchComputeIndirect = (PFNGLDISPATCHCOMPUTEINDIRECTPROC)glewGetProcAddress((const GLubyte*)"glDispatchComputeIndirect")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_compute_shader */

#ifdef GL_ARB_compute_variable_group_size

static GLboolean _glewInit_GL_ARB_compute_variable_group_size (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDispatchComputeGroupSizeARB = (PFNGLDISPATCHCOMPUTEGROUPSIZEARBPROC)glewGetProcAddress((const GLubyte*)"glDispatchComputeGroupSizeARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_compute_variable_group_size */

#ifdef GL_ARB_copy_buffer

static GLboolean _glewInit_GL_ARB_copy_buffer (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCopyBufferSubData = (PFNGLCOPYBUFFERSUBDATAPROC)glewGetProcAddress((const GLubyte*)"glCopyBufferSubData")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_copy_buffer */

#ifdef GL_ARB_copy_image

static GLboolean _glewInit_GL_ARB_copy_image (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCopyImageSubData = (PFNGLCOPYIMAGESUBDATAPROC)glewGetProcAddress((const GLubyte*)"glCopyImageSubData")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_copy_image */

#ifdef GL_ARB_debug_output

static GLboolean _glewInit_GL_ARB_debug_output (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDebugMessageCallbackARB = (PFNGLDEBUGMESSAGECALLBACKARBPROC)glewGetProcAddress((const GLubyte*)"glDebugMessageCallbackARB")) == NULL) || r;
  r = ((glDebugMessageControlARB = (PFNGLDEBUGMESSAGECONTROLARBPROC)glewGetProcAddress((const GLubyte*)"glDebugMessageControlARB")) == NULL) || r;
  r = ((glDebugMessageInsertARB = (PFNGLDEBUGMESSAGEINSERTARBPROC)glewGetProcAddress((const GLubyte*)"glDebugMessageInsertARB")) == NULL) || r;
  r = ((glGetDebugMessageLogARB = (PFNGLGETDEBUGMESSAGELOGARBPROC)glewGetProcAddress((const GLubyte*)"glGetDebugMessageLogARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_debug_output */

#ifdef GL_ARB_direct_state_access

static GLboolean _glewInit_GL_ARB_direct_state_access (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindTextureUnit = (PFNGLBINDTEXTUREUNITPROC)glewGetProcAddress((const GLubyte*)"glBindTextureUnit")) == NULL) || r;
  r = ((glBlitNamedFramebuffer = (PFNGLBLITNAMEDFRAMEBUFFERPROC)glewGetProcAddress((const GLubyte*)"glBlitNamedFramebuffer")) == NULL) || r;
  r = ((glCheckNamedFramebufferStatus = (PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC)glewGetProcAddress((const GLubyte*)"glCheckNamedFramebufferStatus")) == NULL) || r;
  r = ((glClearNamedBufferData = (PFNGLCLEARNAMEDBUFFERDATAPROC)glewGetProcAddress((const GLubyte*)"glClearNamedBufferData")) == NULL) || r;
  r = ((glClearNamedBufferSubData = (PFNGLCLEARNAMEDBUFFERSUBDATAPROC)glewGetProcAddress((const GLubyte*)"glClearNamedBufferSubData")) == NULL) || r;
  r = ((glClearNamedFramebufferfi = (PFNGLCLEARNAMEDFRAMEBUFFERFIPROC)glewGetProcAddress((const GLubyte*)"glClearNamedFramebufferfi")) == NULL) || r;
  r = ((glClearNamedFramebufferfv = (PFNGLCLEARNAMEDFRAMEBUFFERFVPROC)glewGetProcAddress((const GLubyte*)"glClearNamedFramebufferfv")) == NULL) || r;
  r = ((glClearNamedFramebufferiv = (PFNGLCLEARNAMEDFRAMEBUFFERIVPROC)glewGetProcAddress((const GLubyte*)"glClearNamedFramebufferiv")) == NULL) || r;
  r = ((glClearNamedFramebufferuiv = (PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC)glewGetProcAddress((const GLubyte*)"glClearNamedFramebufferuiv")) == NULL) || r;
  r = ((glCompressedTextureSubImage1D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC)glewGetProcAddress((const GLubyte*)"glCompressedTextureSubImage1D")) == NULL) || r;
  r = ((glCompressedTextureSubImage2D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC)glewGetProcAddress((const GLubyte*)"glCompressedTextureSubImage2D")) == NULL) || r;
  r = ((glCompressedTextureSubImage3D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC)glewGetProcAddress((const GLubyte*)"glCompressedTextureSubImage3D")) == NULL) || r;
  r = ((glCopyNamedBufferSubData = (PFNGLCOPYNAMEDBUFFERSUBDATAPROC)glewGetProcAddress((const GLubyte*)"glCopyNamedBufferSubData")) == NULL) || r;
  r = ((glCopyTextureSubImage1D = (PFNGLCOPYTEXTURESUBIMAGE1DPROC)glewGetProcAddress((const GLubyte*)"glCopyTextureSubImage1D")) == NULL) || r;
  r = ((glCopyTextureSubImage2D = (PFNGLCOPYTEXTURESUBIMAGE2DPROC)glewGetProcAddress((const GLubyte*)"glCopyTextureSubImage2D")) == NULL) || r;
  r = ((glCopyTextureSubImage3D = (PFNGLCOPYTEXTURESUBIMAGE3DPROC)glewGetProcAddress((const GLubyte*)"glCopyTextureSubImage3D")) == NULL) || r;
  r = ((glCreateBuffers = (PFNGLCREATEBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glCreateBuffers")) == NULL) || r;
  r = ((glCreateFramebuffers = (PFNGLCREATEFRAMEBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glCreateFramebuffers")) == NULL) || r;
  r = ((glCreateProgramPipelines = (PFNGLCREATEPROGRAMPIPELINESPROC)glewGetProcAddress((const GLubyte*)"glCreateProgramPipelines")) == NULL) || r;
  r = ((glCreateQueries = (PFNGLCREATEQUERIESPROC)glewGetProcAddress((const GLubyte*)"glCreateQueries")) == NULL) || r;
  r = ((glCreateRenderbuffers = (PFNGLCREATERENDERBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glCreateRenderbuffers")) == NULL) || r;
  r = ((glCreateSamplers = (PFNGLCREATESAMPLERSPROC)glewGetProcAddress((const GLubyte*)"glCreateSamplers")) == NULL) || r;
  r = ((glCreateTextures = (PFNGLCREATETEXTURESPROC)glewGetProcAddress((const GLubyte*)"glCreateTextures")) == NULL) || r;
  r = ((glCreateTransformFeedbacks = (PFNGLCREATETRANSFORMFEEDBACKSPROC)glewGetProcAddress((const GLubyte*)"glCreateTransformFeedbacks")) == NULL) || r;
  r = ((glCreateVertexArrays = (PFNGLCREATEVERTEXARRAYSPROC)glewGetProcAddress((const GLubyte*)"glCreateVertexArrays")) == NULL) || r;
  r = ((glDisableVertexArrayAttrib = (PFNGLDISABLEVERTEXARRAYATTRIBPROC)glewGetProcAddress((const GLubyte*)"glDisableVertexArrayAttrib")) == NULL) || r;
  r = ((glEnableVertexArrayAttrib = (PFNGLENABLEVERTEXARRAYATTRIBPROC)glewGetProcAddress((const GLubyte*)"glEnableVertexArrayAttrib")) == NULL) || r;
  r = ((glFlushMappedNamedBufferRange = (PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC)glewGetProcAddress((const GLubyte*)"glFlushMappedNamedBufferRange")) == NULL) || r;
  r = ((glGenerateTextureMipmap = (PFNGLGENERATETEXTUREMIPMAPPROC)glewGetProcAddress((const GLubyte*)"glGenerateTextureMipmap")) == NULL) || r;
  r = ((glGetCompressedTextureImage = (PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC)glewGetProcAddress((const GLubyte*)"glGetCompressedTextureImage")) == NULL) || r;
  r = ((glGetNamedBufferParameteri64v = (PFNGLGETNAMEDBUFFERPARAMETERI64VPROC)glewGetProcAddress((const GLubyte*)"glGetNamedBufferParameteri64v")) == NULL) || r;
  r = ((glGetNamedBufferParameteriv = (PFNGLGETNAMEDBUFFERPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetNamedBufferParameteriv")) == NULL) || r;
  r = ((glGetNamedBufferPointerv = (PFNGLGETNAMEDBUFFERPOINTERVPROC)glewGetProcAddress((const GLubyte*)"glGetNamedBufferPointerv")) == NULL) || r;
  r = ((glGetNamedBufferSubData = (PFNGLGETNAMEDBUFFERSUBDATAPROC)glewGetProcAddress((const GLubyte*)"glGetNamedBufferSubData")) == NULL) || r;
  r = ((glGetNamedFramebufferAttachmentParameteriv = (PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetNamedFramebufferAttachmentParameteriv")) == NULL) || r;
  r = ((glGetNamedFramebufferParameteriv = (PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetNamedFramebufferParameteriv")) == NULL) || r;
  r = ((glGetNamedRenderbufferParameteriv = (PFNGLGETNAMEDRENDERBUFFERPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetNamedRenderbufferParameteriv")) == NULL) || r;
  r = ((glGetQueryBufferObjecti64v = (PFNGLGETQUERYBUFFEROBJECTI64VPROC)glewGetProcAddress((const GLubyte*)"glGetQueryBufferObjecti64v")) == NULL) || r;
  r = ((glGetQueryBufferObjectiv = (PFNGLGETQUERYBUFFEROBJECTIVPROC)glewGetProcAddress((const GLubyte*)"glGetQueryBufferObjectiv")) == NULL) || r;
  r = ((glGetQueryBufferObjectui64v = (PFNGLGETQUERYBUFFEROBJECTUI64VPROC)glewGetProcAddress((const GLubyte*)"glGetQueryBufferObjectui64v")) == NULL) || r;
  r = ((glGetQueryBufferObjectuiv = (PFNGLGETQUERYBUFFEROBJECTUIVPROC)glewGetProcAddress((const GLubyte*)"glGetQueryBufferObjectuiv")) == NULL) || r;
  r = ((glGetTextureImage = (PFNGLGETTEXTUREIMAGEPROC)glewGetProcAddress((const GLubyte*)"glGetTextureImage")) == NULL) || r;
  r = ((glGetTextureLevelParameterfv = (PFNGLGETTEXTURELEVELPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glGetTextureLevelParameterfv")) == NULL) || r;
  r = ((glGetTextureLevelParameteriv = (PFNGLGETTEXTURELEVELPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetTextureLevelParameteriv")) == NULL) || r;
  r = ((glGetTextureParameterIiv = (PFNGLGETTEXTUREPARAMETERIIVPROC)glewGetProcAddress((const GLubyte*)"glGetTextureParameterIiv")) == NULL) || r;
  r = ((glGetTextureParameterIuiv = (PFNGLGETTEXTUREPARAMETERIUIVPROC)glewGetProcAddress((const GLubyte*)"glGetTextureParameterIuiv")) == NULL) || r;
  r = ((glGetTextureParameterfv = (PFNGLGETTEXTUREPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glGetTextureParameterfv")) == NULL) || r;
  r = ((glGetTextureParameteriv = (PFNGLGETTEXTUREPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetTextureParameteriv")) == NULL) || r;
  r = ((glGetTransformFeedbacki64_v = (PFNGLGETTRANSFORMFEEDBACKI64_VPROC)glewGetProcAddress((const GLubyte*)"glGetTransformFeedbacki64_v")) == NULL) || r;
  r = ((glGetTransformFeedbacki_v = (PFNGLGETTRANSFORMFEEDBACKI_VPROC)glewGetProcAddress((const GLubyte*)"glGetTransformFeedbacki_v")) == NULL) || r;
  r = ((glGetTransformFeedbackiv = (PFNGLGETTRANSFORMFEEDBACKIVPROC)glewGetProcAddress((const GLubyte*)"glGetTransformFeedbackiv")) == NULL) || r;
  r = ((glGetVertexArrayIndexed64iv = (PFNGLGETVERTEXARRAYINDEXED64IVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexArrayIndexed64iv")) == NULL) || r;
  r = ((glGetVertexArrayIndexediv = (PFNGLGETVERTEXARRAYINDEXEDIVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexArrayIndexediv")) == NULL) || r;
  r = ((glGetVertexArrayiv = (PFNGLGETVERTEXARRAYIVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexArrayiv")) == NULL) || r;
  r = ((glInvalidateNamedFramebufferData = (PFNGLINVALIDATENAMEDFRAMEBUFFERDATAPROC)glewGetProcAddress((const GLubyte*)"glInvalidateNamedFramebufferData")) == NULL) || r;
  r = ((glInvalidateNamedFramebufferSubData = (PFNGLINVALIDATENAMEDFRAMEBUFFERSUBDATAPROC)glewGetProcAddress((const GLubyte*)"glInvalidateNamedFramebufferSubData")) == NULL) || r;
  r = ((glMapNamedBuffer = (PFNGLMAPNAMEDBUFFERPROC)glewGetProcAddress((const GLubyte*)"glMapNamedBuffer")) == NULL) || r;
  r = ((glMapNamedBufferRange = (PFNGLMAPNAMEDBUFFERRANGEPROC)glewGetProcAddress((const GLubyte*)"glMapNamedBufferRange")) == NULL) || r;
  r = ((glNamedBufferData = (PFNGLNAMEDBUFFERDATAPROC)glewGetProcAddress((const GLubyte*)"glNamedBufferData")) == NULL) || r;
  r = ((glNamedBufferStorage = (PFNGLNAMEDBUFFERSTORAGEPROC)glewGetProcAddress((const GLubyte*)"glNamedBufferStorage")) == NULL) || r;
  r = ((glNamedBufferSubData = (PFNGLNAMEDBUFFERSUBDATAPROC)glewGetProcAddress((const GLubyte*)"glNamedBufferSubData")) == NULL) || r;
  r = ((glNamedFramebufferDrawBuffer = (PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferDrawBuffer")) == NULL) || r;
  r = ((glNamedFramebufferDrawBuffers = (PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferDrawBuffers")) == NULL) || r;
  r = ((glNamedFramebufferParameteri = (PFNGLNAMEDFRAMEBUFFERPARAMETERIPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferParameteri")) == NULL) || r;
  r = ((glNamedFramebufferReadBuffer = (PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferReadBuffer")) == NULL) || r;
  r = ((glNamedFramebufferRenderbuffer = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferRenderbuffer")) == NULL) || r;
  r = ((glNamedFramebufferTexture = (PFNGLNAMEDFRAMEBUFFERTEXTUREPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferTexture")) == NULL) || r;
  r = ((glNamedFramebufferTextureLayer = (PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferTextureLayer")) == NULL) || r;
  r = ((glNamedRenderbufferStorage = (PFNGLNAMEDRENDERBUFFERSTORAGEPROC)glewGetProcAddress((const GLubyte*)"glNamedRenderbufferStorage")) == NULL) || r;
  r = ((glNamedRenderbufferStorageMultisample = (PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEPROC)glewGetProcAddress((const GLubyte*)"glNamedRenderbufferStorageMultisample")) == NULL) || r;
  r = ((glTextureBuffer = (PFNGLTEXTUREBUFFERPROC)glewGetProcAddress((const GLubyte*)"glTextureBuffer")) == NULL) || r;
  r = ((glTextureBufferRange = (PFNGLTEXTUREBUFFERRANGEPROC)glewGetProcAddress((const GLubyte*)"glTextureBufferRange")) == NULL) || r;
  r = ((glTextureParameterIiv = (PFNGLTEXTUREPARAMETERIIVPROC)glewGetProcAddress((const GLubyte*)"glTextureParameterIiv")) == NULL) || r;
  r = ((glTextureParameterIuiv = (PFNGLTEXTUREPARAMETERIUIVPROC)glewGetProcAddress((const GLubyte*)"glTextureParameterIuiv")) == NULL) || r;
  r = ((glTextureParameterf = (PFNGLTEXTUREPARAMETERFPROC)glewGetProcAddress((const GLubyte*)"glTextureParameterf")) == NULL) || r;
  r = ((glTextureParameterfv = (PFNGLTEXTUREPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glTextureParameterfv")) == NULL) || r;
  r = ((glTextureParameteri = (PFNGLTEXTUREPARAMETERIPROC)glewGetProcAddress((const GLubyte*)"glTextureParameteri")) == NULL) || r;
  r = ((glTextureParameteriv = (PFNGLTEXTUREPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glTextureParameteriv")) == NULL) || r;
  r = ((glTextureStorage1D = (PFNGLTEXTURESTORAGE1DPROC)glewGetProcAddress((const GLubyte*)"glTextureStorage1D")) == NULL) || r;
  r = ((glTextureStorage2D = (PFNGLTEXTURESTORAGE2DPROC)glewGetProcAddress((const GLubyte*)"glTextureStorage2D")) == NULL) || r;
  r = ((glTextureStorage2DMultisample = (PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC)glewGetProcAddress((const GLubyte*)"glTextureStorage2DMultisample")) == NULL) || r;
  r = ((glTextureStorage3D = (PFNGLTEXTURESTORAGE3DPROC)glewGetProcAddress((const GLubyte*)"glTextureStorage3D")) == NULL) || r;
  r = ((glTextureStorage3DMultisample = (PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC)glewGetProcAddress((const GLubyte*)"glTextureStorage3DMultisample")) == NULL) || r;
  r = ((glTextureSubImage1D = (PFNGLTEXTURESUBIMAGE1DPROC)glewGetProcAddress((const GLubyte*)"glTextureSubImage1D")) == NULL) || r;
  r = ((glTextureSubImage2D = (PFNGLTEXTURESUBIMAGE2DPROC)glewGetProcAddress((const GLubyte*)"glTextureSubImage2D")) == NULL) || r;
  r = ((glTextureSubImage3D = (PFNGLTEXTURESUBIMAGE3DPROC)glewGetProcAddress((const GLubyte*)"glTextureSubImage3D")) == NULL) || r;
  r = ((glTransformFeedbackBufferBase = (PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC)glewGetProcAddress((const GLubyte*)"glTransformFeedbackBufferBase")) == NULL) || r;
  r = ((glTransformFeedbackBufferRange = (PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC)glewGetProcAddress((const GLubyte*)"glTransformFeedbackBufferRange")) == NULL) || r;
  r = ((glUnmapNamedBuffer = (PFNGLUNMAPNAMEDBUFFERPROC)glewGetProcAddress((const GLubyte*)"glUnmapNamedBuffer")) == NULL) || r;
  r = ((glVertexArrayAttribBinding = (PFNGLVERTEXARRAYATTRIBBINDINGPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayAttribBinding")) == NULL) || r;
  r = ((glVertexArrayAttribFormat = (PFNGLVERTEXARRAYATTRIBFORMATPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayAttribFormat")) == NULL) || r;
  r = ((glVertexArrayAttribIFormat = (PFNGLVERTEXARRAYATTRIBIFORMATPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayAttribIFormat")) == NULL) || r;
  r = ((glVertexArrayAttribLFormat = (PFNGLVERTEXARRAYATTRIBLFORMATPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayAttribLFormat")) == NULL) || r;
  r = ((glVertexArrayBindingDivisor = (PFNGLVERTEXARRAYBINDINGDIVISORPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayBindingDivisor")) == NULL) || r;
  r = ((glVertexArrayElementBuffer = (PFNGLVERTEXARRAYELEMENTBUFFERPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayElementBuffer")) == NULL) || r;
  r = ((glVertexArrayVertexBuffer = (PFNGLVERTEXARRAYVERTEXBUFFERPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexBuffer")) == NULL) || r;
  r = ((glVertexArrayVertexBuffers = (PFNGLVERTEXARRAYVERTEXBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexBuffers")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_direct_state_access */

#ifdef GL_ARB_draw_buffers

static GLboolean _glewInit_GL_ARB_draw_buffers (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawBuffersARB = (PFNGLDRAWBUFFERSARBPROC)glewGetProcAddress((const GLubyte*)"glDrawBuffersARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_draw_buffers */

#ifdef GL_ARB_draw_buffers_blend

static GLboolean _glewInit_GL_ARB_draw_buffers_blend (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendEquationSeparateiARB = (PFNGLBLENDEQUATIONSEPARATEIARBPROC)glewGetProcAddress((const GLubyte*)"glBlendEquationSeparateiARB")) == NULL) || r;
  r = ((glBlendEquationiARB = (PFNGLBLENDEQUATIONIARBPROC)glewGetProcAddress((const GLubyte*)"glBlendEquationiARB")) == NULL) || r;
  r = ((glBlendFuncSeparateiARB = (PFNGLBLENDFUNCSEPARATEIARBPROC)glewGetProcAddress((const GLubyte*)"glBlendFuncSeparateiARB")) == NULL) || r;
  r = ((glBlendFunciARB = (PFNGLBLENDFUNCIARBPROC)glewGetProcAddress((const GLubyte*)"glBlendFunciARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_draw_buffers_blend */

#ifdef GL_ARB_draw_elements_base_vertex

static GLboolean _glewInit_GL_ARB_draw_elements_base_vertex (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawElementsBaseVertex = (PFNGLDRAWELEMENTSBASEVERTEXPROC)glewGetProcAddress((const GLubyte*)"glDrawElementsBaseVertex")) == NULL) || r;
  r = ((glDrawElementsInstancedBaseVertex = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC)glewGetProcAddress((const GLubyte*)"glDrawElementsInstancedBaseVertex")) == NULL) || r;
  r = ((glDrawRangeElementsBaseVertex = (PFNGLDRAWRANGEELEMENTSBASEVERTEXPROC)glewGetProcAddress((const GLubyte*)"glDrawRangeElementsBaseVertex")) == NULL) || r;
  r = ((glMultiDrawElementsBaseVertex = (PFNGLMULTIDRAWELEMENTSBASEVERTEXPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawElementsBaseVertex")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_draw_elements_base_vertex */

#ifdef GL_ARB_draw_indirect

static GLboolean _glewInit_GL_ARB_draw_indirect (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawArraysIndirect = (PFNGLDRAWARRAYSINDIRECTPROC)glewGetProcAddress((const GLubyte*)"glDrawArraysIndirect")) == NULL) || r;
  r = ((glDrawElementsIndirect = (PFNGLDRAWELEMENTSINDIRECTPROC)glewGetProcAddress((const GLubyte*)"glDrawElementsIndirect")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_draw_indirect */

#ifdef GL_ARB_framebuffer_no_attachments

static GLboolean _glewInit_GL_ARB_framebuffer_no_attachments (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFramebufferParameteri = (PFNGLFRAMEBUFFERPARAMETERIPROC)glewGetProcAddress((const GLubyte*)"glFramebufferParameteri")) == NULL) || r;
  r = ((glGetFramebufferParameteriv = (PFNGLGETFRAMEBUFFERPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetFramebufferParameteriv")) == NULL) || r;
  r = ((glGetNamedFramebufferParameterivEXT = (PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedFramebufferParameterivEXT")) == NULL) || r;
  r = ((glNamedFramebufferParameteriEXT = (PFNGLNAMEDFRAMEBUFFERPARAMETERIEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferParameteriEXT")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_framebuffer_no_attachments */

#ifdef GL_ARB_framebuffer_object

static GLboolean _glewInit_GL_ARB_framebuffer_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC)glewGetProcAddress((const GLubyte*)"glBindFramebuffer")) == NULL) || r;
  r = ((glBindRenderbuffer = (PFNGLBINDRENDERBUFFERPROC)glewGetProcAddress((const GLubyte*)"glBindRenderbuffer")) == NULL) || r;
  r = ((glBlitFramebuffer = (PFNGLBLITFRAMEBUFFERPROC)glewGetProcAddress((const GLubyte*)"glBlitFramebuffer")) == NULL) || r;
  r = ((glCheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUSPROC)glewGetProcAddress((const GLubyte*)"glCheckFramebufferStatus")) == NULL) || r;
  r = ((glDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glDeleteFramebuffers")) == NULL) || r;
  r = ((glDeleteRenderbuffers = (PFNGLDELETERENDERBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glDeleteRenderbuffers")) == NULL) || r;
  r = ((glFramebufferRenderbuffer = (PFNGLFRAMEBUFFERRENDERBUFFERPROC)glewGetProcAddress((const GLubyte*)"glFramebufferRenderbuffer")) == NULL) || r;
  r = ((glFramebufferTexture1D = (PFNGLFRAMEBUFFERTEXTURE1DPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTexture1D")) == NULL) || r;
  r = ((glFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTexture2D")) == NULL) || r;
  r = ((glFramebufferTexture3D = (PFNGLFRAMEBUFFERTEXTURE3DPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTexture3D")) == NULL) || r;
  r = ((glFramebufferTextureLayer = (PFNGLFRAMEBUFFERTEXTURELAYERPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTextureLayer")) == NULL) || r;
  r = ((glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glGenFramebuffers")) == NULL) || r;
  r = ((glGenRenderbuffers = (PFNGLGENRENDERBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glGenRenderbuffers")) == NULL) || r;
  r = ((glGenerateMipmap = (PFNGLGENERATEMIPMAPPROC)glewGetProcAddress((const GLubyte*)"glGenerateMipmap")) == NULL) || r;
  r = ((glGetFramebufferAttachmentParameteriv = (PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetFramebufferAttachmentParameteriv")) == NULL) || r;
  r = ((glGetRenderbufferParameteriv = (PFNGLGETRENDERBUFFERPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetRenderbufferParameteriv")) == NULL) || r;
  r = ((glIsFramebuffer = (PFNGLISFRAMEBUFFERPROC)glewGetProcAddress((const GLubyte*)"glIsFramebuffer")) == NULL) || r;
  r = ((glIsRenderbuffer = (PFNGLISRENDERBUFFERPROC)glewGetProcAddress((const GLubyte*)"glIsRenderbuffer")) == NULL) || r;
  r = ((glRenderbufferStorage = (PFNGLRENDERBUFFERSTORAGEPROC)glewGetProcAddress((const GLubyte*)"glRenderbufferStorage")) == NULL) || r;
  r = ((glRenderbufferStorageMultisample = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC)glewGetProcAddress((const GLubyte*)"glRenderbufferStorageMultisample")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_framebuffer_object */

#ifdef GL_ARB_geometry_shader4

static GLboolean _glewInit_GL_ARB_geometry_shader4 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFramebufferTextureARB = (PFNGLFRAMEBUFFERTEXTUREARBPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTextureARB")) == NULL) || r;
  r = ((glFramebufferTextureFaceARB = (PFNGLFRAMEBUFFERTEXTUREFACEARBPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTextureFaceARB")) == NULL) || r;
  r = ((glFramebufferTextureLayerARB = (PFNGLFRAMEBUFFERTEXTURELAYERARBPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTextureLayerARB")) == NULL) || r;
  r = ((glProgramParameteriARB = (PFNGLPROGRAMPARAMETERIARBPROC)glewGetProcAddress((const GLubyte*)"glProgramParameteriARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_geometry_shader4 */

#ifdef GL_ARB_get_program_binary

static GLboolean _glewInit_GL_ARB_get_program_binary (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetProgramBinary = (PFNGLGETPROGRAMBINARYPROC)glewGetProcAddress((const GLubyte*)"glGetProgramBinary")) == NULL) || r;
  r = ((glProgramBinary = (PFNGLPROGRAMBINARYPROC)glewGetProcAddress((const GLubyte*)"glProgramBinary")) == NULL) || r;
  r = ((glProgramParameteri = (PFNGLPROGRAMPARAMETERIPROC)glewGetProcAddress((const GLubyte*)"glProgramParameteri")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_get_program_binary */

#ifdef GL_ARB_get_texture_sub_image

static GLboolean _glewInit_GL_ARB_get_texture_sub_image (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetCompressedTextureSubImage = (PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC)glewGetProcAddress((const GLubyte*)"glGetCompressedTextureSubImage")) == NULL) || r;
  r = ((glGetTextureSubImage = (PFNGLGETTEXTURESUBIMAGEPROC)glewGetProcAddress((const GLubyte*)"glGetTextureSubImage")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_get_texture_sub_image */

#ifdef GL_ARB_gpu_shader_fp64

static GLboolean _glewInit_GL_ARB_gpu_shader_fp64 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetUniformdv = (PFNGLGETUNIFORMDVPROC)glewGetProcAddress((const GLubyte*)"glGetUniformdv")) == NULL) || r;
  r = ((glUniform1d = (PFNGLUNIFORM1DPROC)glewGetProcAddress((const GLubyte*)"glUniform1d")) == NULL) || r;
  r = ((glUniform1dv = (PFNGLUNIFORM1DVPROC)glewGetProcAddress((const GLubyte*)"glUniform1dv")) == NULL) || r;
  r = ((glUniform2d = (PFNGLUNIFORM2DPROC)glewGetProcAddress((const GLubyte*)"glUniform2d")) == NULL) || r;
  r = ((glUniform2dv = (PFNGLUNIFORM2DVPROC)glewGetProcAddress((const GLubyte*)"glUniform2dv")) == NULL) || r;
  r = ((glUniform3d = (PFNGLUNIFORM3DPROC)glewGetProcAddress((const GLubyte*)"glUniform3d")) == NULL) || r;
  r = ((glUniform3dv = (PFNGLUNIFORM3DVPROC)glewGetProcAddress((const GLubyte*)"glUniform3dv")) == NULL) || r;
  r = ((glUniform4d = (PFNGLUNIFORM4DPROC)glewGetProcAddress((const GLubyte*)"glUniform4d")) == NULL) || r;
  r = ((glUniform4dv = (PFNGLUNIFORM4DVPROC)glewGetProcAddress((const GLubyte*)"glUniform4dv")) == NULL) || r;
  r = ((glUniformMatrix2dv = (PFNGLUNIFORMMATRIX2DVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix2dv")) == NULL) || r;
  r = ((glUniformMatrix2x3dv = (PFNGLUNIFORMMATRIX2X3DVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix2x3dv")) == NULL) || r;
  r = ((glUniformMatrix2x4dv = (PFNGLUNIFORMMATRIX2X4DVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix2x4dv")) == NULL) || r;
  r = ((glUniformMatrix3dv = (PFNGLUNIFORMMATRIX3DVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix3dv")) == NULL) || r;
  r = ((glUniformMatrix3x2dv = (PFNGLUNIFORMMATRIX3X2DVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix3x2dv")) == NULL) || r;
  r = ((glUniformMatrix3x4dv = (PFNGLUNIFORMMATRIX3X4DVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix3x4dv")) == NULL) || r;
  r = ((glUniformMatrix4dv = (PFNGLUNIFORMMATRIX4DVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix4dv")) == NULL) || r;
  r = ((glUniformMatrix4x2dv = (PFNGLUNIFORMMATRIX4X2DVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix4x2dv")) == NULL) || r;
  r = ((glUniformMatrix4x3dv = (PFNGLUNIFORMMATRIX4X3DVPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix4x3dv")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_gpu_shader_fp64 */

#ifdef GL_ARB_gpu_shader_int64

static GLboolean _glewInit_GL_ARB_gpu_shader_int64 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetUniformi64vARB = (PFNGLGETUNIFORMI64VARBPROC)glewGetProcAddress((const GLubyte*)"glGetUniformi64vARB")) == NULL) || r;
  r = ((glGetUniformui64vARB = (PFNGLGETUNIFORMUI64VARBPROC)glewGetProcAddress((const GLubyte*)"glGetUniformui64vARB")) == NULL) || r;
  r = ((glGetnUniformi64vARB = (PFNGLGETNUNIFORMI64VARBPROC)glewGetProcAddress((const GLubyte*)"glGetnUniformi64vARB")) == NULL) || r;
  r = ((glGetnUniformui64vARB = (PFNGLGETNUNIFORMUI64VARBPROC)glewGetProcAddress((const GLubyte*)"glGetnUniformui64vARB")) == NULL) || r;
  r = ((glProgramUniform1i64ARB = (PFNGLPROGRAMUNIFORM1I64ARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1i64ARB")) == NULL) || r;
  r = ((glProgramUniform1i64vARB = (PFNGLPROGRAMUNIFORM1I64VARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1i64vARB")) == NULL) || r;
  r = ((glProgramUniform1ui64ARB = (PFNGLPROGRAMUNIFORM1UI64ARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1ui64ARB")) == NULL) || r;
  r = ((glProgramUniform1ui64vARB = (PFNGLPROGRAMUNIFORM1UI64VARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1ui64vARB")) == NULL) || r;
  r = ((glProgramUniform2i64ARB = (PFNGLPROGRAMUNIFORM2I64ARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2i64ARB")) == NULL) || r;
  r = ((glProgramUniform2i64vARB = (PFNGLPROGRAMUNIFORM2I64VARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2i64vARB")) == NULL) || r;
  r = ((glProgramUniform2ui64ARB = (PFNGLPROGRAMUNIFORM2UI64ARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2ui64ARB")) == NULL) || r;
  r = ((glProgramUniform2ui64vARB = (PFNGLPROGRAMUNIFORM2UI64VARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2ui64vARB")) == NULL) || r;
  r = ((glProgramUniform3i64ARB = (PFNGLPROGRAMUNIFORM3I64ARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3i64ARB")) == NULL) || r;
  r = ((glProgramUniform3i64vARB = (PFNGLPROGRAMUNIFORM3I64VARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3i64vARB")) == NULL) || r;
  r = ((glProgramUniform3ui64ARB = (PFNGLPROGRAMUNIFORM3UI64ARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3ui64ARB")) == NULL) || r;
  r = ((glProgramUniform3ui64vARB = (PFNGLPROGRAMUNIFORM3UI64VARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3ui64vARB")) == NULL) || r;
  r = ((glProgramUniform4i64ARB = (PFNGLPROGRAMUNIFORM4I64ARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4i64ARB")) == NULL) || r;
  r = ((glProgramUniform4i64vARB = (PFNGLPROGRAMUNIFORM4I64VARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4i64vARB")) == NULL) || r;
  r = ((glProgramUniform4ui64ARB = (PFNGLPROGRAMUNIFORM4UI64ARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4ui64ARB")) == NULL) || r;
  r = ((glProgramUniform4ui64vARB = (PFNGLPROGRAMUNIFORM4UI64VARBPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4ui64vARB")) == NULL) || r;
  r = ((glUniform1i64ARB = (PFNGLUNIFORM1I64ARBPROC)glewGetProcAddress((const GLubyte*)"glUniform1i64ARB")) == NULL) || r;
  r = ((glUniform1i64vARB = (PFNGLUNIFORM1I64VARBPROC)glewGetProcAddress((const GLubyte*)"glUniform1i64vARB")) == NULL) || r;
  r = ((glUniform1ui64ARB = (PFNGLUNIFORM1UI64ARBPROC)glewGetProcAddress((const GLubyte*)"glUniform1ui64ARB")) == NULL) || r;
  r = ((glUniform1ui64vARB = (PFNGLUNIFORM1UI64VARBPROC)glewGetProcAddress((const GLubyte*)"glUniform1ui64vARB")) == NULL) || r;
  r = ((glUniform2i64ARB = (PFNGLUNIFORM2I64ARBPROC)glewGetProcAddress((const GLubyte*)"glUniform2i64ARB")) == NULL) || r;
  r = ((glUniform2i64vARB = (PFNGLUNIFORM2I64VARBPROC)glewGetProcAddress((const GLubyte*)"glUniform2i64vARB")) == NULL) || r;
  r = ((glUniform2ui64ARB = (PFNGLUNIFORM2UI64ARBPROC)glewGetProcAddress((const GLubyte*)"glUniform2ui64ARB")) == NULL) || r;
  r = ((glUniform2ui64vARB = (PFNGLUNIFORM2UI64VARBPROC)glewGetProcAddress((const GLubyte*)"glUniform2ui64vARB")) == NULL) || r;
  r = ((glUniform3i64ARB = (PFNGLUNIFORM3I64ARBPROC)glewGetProcAddress((const GLubyte*)"glUniform3i64ARB")) == NULL) || r;
  r = ((glUniform3i64vARB = (PFNGLUNIFORM3I64VARBPROC)glewGetProcAddress((const GLubyte*)"glUniform3i64vARB")) == NULL) || r;
  r = ((glUniform3ui64ARB = (PFNGLUNIFORM3UI64ARBPROC)glewGetProcAddress((const GLubyte*)"glUniform3ui64ARB")) == NULL) || r;
  r = ((glUniform3ui64vARB = (PFNGLUNIFORM3UI64VARBPROC)glewGetProcAddress((const GLubyte*)"glUniform3ui64vARB")) == NULL) || r;
  r = ((glUniform4i64ARB = (PFNGLUNIFORM4I64ARBPROC)glewGetProcAddress((const GLubyte*)"glUniform4i64ARB")) == NULL) || r;
  r = ((glUniform4i64vARB = (PFNGLUNIFORM4I64VARBPROC)glewGetProcAddress((const GLubyte*)"glUniform4i64vARB")) == NULL) || r;
  r = ((glUniform4ui64ARB = (PFNGLUNIFORM4UI64ARBPROC)glewGetProcAddress((const GLubyte*)"glUniform4ui64ARB")) == NULL) || r;
  r = ((glUniform4ui64vARB = (PFNGLUNIFORM4UI64VARBPROC)glewGetProcAddress((const GLubyte*)"glUniform4ui64vARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_gpu_shader_int64 */

#ifdef GL_ARB_imaging

static GLboolean _glewInit_GL_ARB_imaging (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendEquation = (PFNGLBLENDEQUATIONPROC)glewGetProcAddress((const GLubyte*)"glBlendEquation")) == NULL) || r;
  r = ((glColorSubTable = (PFNGLCOLORSUBTABLEPROC)glewGetProcAddress((const GLubyte*)"glColorSubTable")) == NULL) || r;
  r = ((glColorTable = (PFNGLCOLORTABLEPROC)glewGetProcAddress((const GLubyte*)"glColorTable")) == NULL) || r;
  r = ((glColorTableParameterfv = (PFNGLCOLORTABLEPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glColorTableParameterfv")) == NULL) || r;
  r = ((glColorTableParameteriv = (PFNGLCOLORTABLEPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glColorTableParameteriv")) == NULL) || r;
  r = ((glConvolutionFilter1D = (PFNGLCONVOLUTIONFILTER1DPROC)glewGetProcAddress((const GLubyte*)"glConvolutionFilter1D")) == NULL) || r;
  r = ((glConvolutionFilter2D = (PFNGLCONVOLUTIONFILTER2DPROC)glewGetProcAddress((const GLubyte*)"glConvolutionFilter2D")) == NULL) || r;
  r = ((glConvolutionParameterf = (PFNGLCONVOLUTIONPARAMETERFPROC)glewGetProcAddress((const GLubyte*)"glConvolutionParameterf")) == NULL) || r;
  r = ((glConvolutionParameterfv = (PFNGLCONVOLUTIONPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glConvolutionParameterfv")) == NULL) || r;
  r = ((glConvolutionParameteri = (PFNGLCONVOLUTIONPARAMETERIPROC)glewGetProcAddress((const GLubyte*)"glConvolutionParameteri")) == NULL) || r;
  r = ((glConvolutionParameteriv = (PFNGLCONVOLUTIONPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glConvolutionParameteriv")) == NULL) || r;
  r = ((glCopyColorSubTable = (PFNGLCOPYCOLORSUBTABLEPROC)glewGetProcAddress((const GLubyte*)"glCopyColorSubTable")) == NULL) || r;
  r = ((glCopyColorTable = (PFNGLCOPYCOLORTABLEPROC)glewGetProcAddress((const GLubyte*)"glCopyColorTable")) == NULL) || r;
  r = ((glCopyConvolutionFilter1D = (PFNGLCOPYCONVOLUTIONFILTER1DPROC)glewGetProcAddress((const GLubyte*)"glCopyConvolutionFilter1D")) == NULL) || r;
  r = ((glCopyConvolutionFilter2D = (PFNGLCOPYCONVOLUTIONFILTER2DPROC)glewGetProcAddress((const GLubyte*)"glCopyConvolutionFilter2D")) == NULL) || r;
  r = ((glGetColorTable = (PFNGLGETCOLORTABLEPROC)glewGetProcAddress((const GLubyte*)"glGetColorTable")) == NULL) || r;
  r = ((glGetColorTableParameterfv = (PFNGLGETCOLORTABLEPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glGetColorTableParameterfv")) == NULL) || r;
  r = ((glGetColorTableParameteriv = (PFNGLGETCOLORTABLEPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetColorTableParameteriv")) == NULL) || r;
  r = ((glGetConvolutionFilter = (PFNGLGETCONVOLUTIONFILTERPROC)glewGetProcAddress((const GLubyte*)"glGetConvolutionFilter")) == NULL) || r;
  r = ((glGetConvolutionParameterfv = (PFNGLGETCONVOLUTIONPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glGetConvolutionParameterfv")) == NULL) || r;
  r = ((glGetConvolutionParameteriv = (PFNGLGETCONVOLUTIONPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetConvolutionParameteriv")) == NULL) || r;
  r = ((glGetHistogram = (PFNGLGETHISTOGRAMPROC)glewGetProcAddress((const GLubyte*)"glGetHistogram")) == NULL) || r;
  r = ((glGetHistogramParameterfv = (PFNGLGETHISTOGRAMPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glGetHistogramParameterfv")) == NULL) || r;
  r = ((glGetHistogramParameteriv = (PFNGLGETHISTOGRAMPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetHistogramParameteriv")) == NULL) || r;
  r = ((glGetMinmax = (PFNGLGETMINMAXPROC)glewGetProcAddress((const GLubyte*)"glGetMinmax")) == NULL) || r;
  r = ((glGetMinmaxParameterfv = (PFNGLGETMINMAXPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glGetMinmaxParameterfv")) == NULL) || r;
  r = ((glGetMinmaxParameteriv = (PFNGLGETMINMAXPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetMinmaxParameteriv")) == NULL) || r;
  r = ((glGetSeparableFilter = (PFNGLGETSEPARABLEFILTERPROC)glewGetProcAddress((const GLubyte*)"glGetSeparableFilter")) == NULL) || r;
  r = ((glHistogram = (PFNGLHISTOGRAMPROC)glewGetProcAddress((const GLubyte*)"glHistogram")) == NULL) || r;
  r = ((glMinmax = (PFNGLMINMAXPROC)glewGetProcAddress((const GLubyte*)"glMinmax")) == NULL) || r;
  r = ((glResetHistogram = (PFNGLRESETHISTOGRAMPROC)glewGetProcAddress((const GLubyte*)"glResetHistogram")) == NULL) || r;
  r = ((glResetMinmax = (PFNGLRESETMINMAXPROC)glewGetProcAddress((const GLubyte*)"glResetMinmax")) == NULL) || r;
  r = ((glSeparableFilter2D = (PFNGLSEPARABLEFILTER2DPROC)glewGetProcAddress((const GLubyte*)"glSeparableFilter2D")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_imaging */

#ifdef GL_ARB_indirect_parameters

static GLboolean _glewInit_GL_ARB_indirect_parameters (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMultiDrawArraysIndirectCountARB = (PFNGLMULTIDRAWARRAYSINDIRECTCOUNTARBPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawArraysIndirectCountARB")) == NULL) || r;
  r = ((glMultiDrawElementsIndirectCountARB = (PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTARBPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawElementsIndirectCountARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_indirect_parameters */

#ifdef GL_ARB_instanced_arrays

static GLboolean _glewInit_GL_ARB_instanced_arrays (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawArraysInstancedARB = (PFNGLDRAWARRAYSINSTANCEDARBPROC)glewGetProcAddress((const GLubyte*)"glDrawArraysInstancedARB")) == NULL) || r;
  r = ((glDrawElementsInstancedARB = (PFNGLDRAWELEMENTSINSTANCEDARBPROC)glewGetProcAddress((const GLubyte*)"glDrawElementsInstancedARB")) == NULL) || r;
  r = ((glVertexAttribDivisorARB = (PFNGLVERTEXATTRIBDIVISORARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribDivisorARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_instanced_arrays */

#ifdef GL_ARB_internalformat_query

static GLboolean _glewInit_GL_ARB_internalformat_query (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetInternalformativ = (PFNGLGETINTERNALFORMATIVPROC)glewGetProcAddress((const GLubyte*)"glGetInternalformativ")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_internalformat_query */

#ifdef GL_ARB_internalformat_query2

static GLboolean _glewInit_GL_ARB_internalformat_query2 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetInternalformati64v = (PFNGLGETINTERNALFORMATI64VPROC)glewGetProcAddress((const GLubyte*)"glGetInternalformati64v")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_internalformat_query2 */

#ifdef GL_ARB_invalidate_subdata

static GLboolean _glewInit_GL_ARB_invalidate_subdata (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glInvalidateBufferData = (PFNGLINVALIDATEBUFFERDATAPROC)glewGetProcAddress((const GLubyte*)"glInvalidateBufferData")) == NULL) || r;
  r = ((glInvalidateBufferSubData = (PFNGLINVALIDATEBUFFERSUBDATAPROC)glewGetProcAddress((const GLubyte*)"glInvalidateBufferSubData")) == NULL) || r;
  r = ((glInvalidateFramebuffer = (PFNGLINVALIDATEFRAMEBUFFERPROC)glewGetProcAddress((const GLubyte*)"glInvalidateFramebuffer")) == NULL) || r;
  r = ((glInvalidateSubFramebuffer = (PFNGLINVALIDATESUBFRAMEBUFFERPROC)glewGetProcAddress((const GLubyte*)"glInvalidateSubFramebuffer")) == NULL) || r;
  r = ((glInvalidateTexImage = (PFNGLINVALIDATETEXIMAGEPROC)glewGetProcAddress((const GLubyte*)"glInvalidateTexImage")) == NULL) || r;
  r = ((glInvalidateTexSubImage = (PFNGLINVALIDATETEXSUBIMAGEPROC)glewGetProcAddress((const GLubyte*)"glInvalidateTexSubImage")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_invalidate_subdata */

#ifdef GL_ARB_map_buffer_range

static GLboolean _glewInit_GL_ARB_map_buffer_range (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFlushMappedBufferRange = (PFNGLFLUSHMAPPEDBUFFERRANGEPROC)glewGetProcAddress((const GLubyte*)"glFlushMappedBufferRange")) == NULL) || r;
  r = ((glMapBufferRange = (PFNGLMAPBUFFERRANGEPROC)glewGetProcAddress((const GLubyte*)"glMapBufferRange")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_map_buffer_range */

#ifdef GL_ARB_matrix_palette

static GLboolean _glewInit_GL_ARB_matrix_palette (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCurrentPaletteMatrixARB = (PFNGLCURRENTPALETTEMATRIXARBPROC)glewGetProcAddress((const GLubyte*)"glCurrentPaletteMatrixARB")) == NULL) || r;
  r = ((glMatrixIndexPointerARB = (PFNGLMATRIXINDEXPOINTERARBPROC)glewGetProcAddress((const GLubyte*)"glMatrixIndexPointerARB")) == NULL) || r;
  r = ((glMatrixIndexubvARB = (PFNGLMATRIXINDEXUBVARBPROC)glewGetProcAddress((const GLubyte*)"glMatrixIndexubvARB")) == NULL) || r;
  r = ((glMatrixIndexuivARB = (PFNGLMATRIXINDEXUIVARBPROC)glewGetProcAddress((const GLubyte*)"glMatrixIndexuivARB")) == NULL) || r;
  r = ((glMatrixIndexusvARB = (PFNGLMATRIXINDEXUSVARBPROC)glewGetProcAddress((const GLubyte*)"glMatrixIndexusvARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_matrix_palette */

#ifdef GL_ARB_multi_bind

static GLboolean _glewInit_GL_ARB_multi_bind (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindBuffersBase = (PFNGLBINDBUFFERSBASEPROC)glewGetProcAddress((const GLubyte*)"glBindBuffersBase")) == NULL) || r;
  r = ((glBindBuffersRange = (PFNGLBINDBUFFERSRANGEPROC)glewGetProcAddress((const GLubyte*)"glBindBuffersRange")) == NULL) || r;
  r = ((glBindImageTextures = (PFNGLBINDIMAGETEXTURESPROC)glewGetProcAddress((const GLubyte*)"glBindImageTextures")) == NULL) || r;
  r = ((glBindSamplers = (PFNGLBINDSAMPLERSPROC)glewGetProcAddress((const GLubyte*)"glBindSamplers")) == NULL) || r;
  r = ((glBindTextures = (PFNGLBINDTEXTURESPROC)glewGetProcAddress((const GLubyte*)"glBindTextures")) == NULL) || r;
  r = ((glBindVertexBuffers = (PFNGLBINDVERTEXBUFFERSPROC)glewGetProcAddress((const GLubyte*)"glBindVertexBuffers")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_multi_bind */

#ifdef GL_ARB_multi_draw_indirect

static GLboolean _glewInit_GL_ARB_multi_draw_indirect (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMultiDrawArraysIndirect = (PFNGLMULTIDRAWARRAYSINDIRECTPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawArraysIndirect")) == NULL) || r;
  r = ((glMultiDrawElementsIndirect = (PFNGLMULTIDRAWELEMENTSINDIRECTPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawElementsIndirect")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_multi_draw_indirect */

#ifdef GL_ARB_multisample

static GLboolean _glewInit_GL_ARB_multisample (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glSampleCoverageARB = (PFNGLSAMPLECOVERAGEARBPROC)glewGetProcAddress((const GLubyte*)"glSampleCoverageARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_multisample */

#ifdef GL_ARB_multitexture

static GLboolean _glewInit_GL_ARB_multitexture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC)glewGetProcAddress((const GLubyte*)"glActiveTextureARB")) == NULL) || r;
  r = ((glClientActiveTextureARB = (PFNGLCLIENTACTIVETEXTUREARBPROC)glewGetProcAddress((const GLubyte*)"glClientActiveTextureARB")) == NULL) || r;
  r = ((glMultiTexCoord1dARB = (PFNGLMULTITEXCOORD1DARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1dARB")) == NULL) || r;
  r = ((glMultiTexCoord1dvARB = (PFNGLMULTITEXCOORD1DVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1dvARB")) == NULL) || r;
  r = ((glMultiTexCoord1fARB = (PFNGLMULTITEXCOORD1FARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1fARB")) == NULL) || r;
  r = ((glMultiTexCoord1fvARB = (PFNGLMULTITEXCOORD1FVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1fvARB")) == NULL) || r;
  r = ((glMultiTexCoord1iARB = (PFNGLMULTITEXCOORD1IARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1iARB")) == NULL) || r;
  r = ((glMultiTexCoord1ivARB = (PFNGLMULTITEXCOORD1IVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1ivARB")) == NULL) || r;
  r = ((glMultiTexCoord1sARB = (PFNGLMULTITEXCOORD1SARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1sARB")) == NULL) || r;
  r = ((glMultiTexCoord1svARB = (PFNGLMULTITEXCOORD1SVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1svARB")) == NULL) || r;
  r = ((glMultiTexCoord2dARB = (PFNGLMULTITEXCOORD2DARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2dARB")) == NULL) || r;
  r = ((glMultiTexCoord2dvARB = (PFNGLMULTITEXCOORD2DVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2dvARB")) == NULL) || r;
  r = ((glMultiTexCoord2fARB = (PFNGLMULTITEXCOORD2FARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2fARB")) == NULL) || r;
  r = ((glMultiTexCoord2fvARB = (PFNGLMULTITEXCOORD2FVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2fvARB")) == NULL) || r;
  r = ((glMultiTexCoord2iARB = (PFNGLMULTITEXCOORD2IARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2iARB")) == NULL) || r;
  r = ((glMultiTexCoord2ivARB = (PFNGLMULTITEXCOORD2IVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2ivARB")) == NULL) || r;
  r = ((glMultiTexCoord2sARB = (PFNGLMULTITEXCOORD2SARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2sARB")) == NULL) || r;
  r = ((glMultiTexCoord2svARB = (PFNGLMULTITEXCOORD2SVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2svARB")) == NULL) || r;
  r = ((glMultiTexCoord3dARB = (PFNGLMULTITEXCOORD3DARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3dARB")) == NULL) || r;
  r = ((glMultiTexCoord3dvARB = (PFNGLMULTITEXCOORD3DVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3dvARB")) == NULL) || r;
  r = ((glMultiTexCoord3fARB = (PFNGLMULTITEXCOORD3FARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3fARB")) == NULL) || r;
  r = ((glMultiTexCoord3fvARB = (PFNGLMULTITEXCOORD3FVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3fvARB")) == NULL) || r;
  r = ((glMultiTexCoord3iARB = (PFNGLMULTITEXCOORD3IARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3iARB")) == NULL) || r;
  r = ((glMultiTexCoord3ivARB = (PFNGLMULTITEXCOORD3IVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3ivARB")) == NULL) || r;
  r = ((glMultiTexCoord3sARB = (PFNGLMULTITEXCOORD3SARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3sARB")) == NULL) || r;
  r = ((glMultiTexCoord3svARB = (PFNGLMULTITEXCOORD3SVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3svARB")) == NULL) || r;
  r = ((glMultiTexCoord4dARB = (PFNGLMULTITEXCOORD4DARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4dARB")) == NULL) || r;
  r = ((glMultiTexCoord4dvARB = (PFNGLMULTITEXCOORD4DVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4dvARB")) == NULL) || r;
  r = ((glMultiTexCoord4fARB = (PFNGLMULTITEXCOORD4FARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4fARB")) == NULL) || r;
  r = ((glMultiTexCoord4fvARB = (PFNGLMULTITEXCOORD4FVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4fvARB")) == NULL) || r;
  r = ((glMultiTexCoord4iARB = (PFNGLMULTITEXCOORD4IARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4iARB")) == NULL) || r;
  r = ((glMultiTexCoord4ivARB = (PFNGLMULTITEXCOORD4IVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4ivARB")) == NULL) || r;
  r = ((glMultiTexCoord4sARB = (PFNGLMULTITEXCOORD4SARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4sARB")) == NULL) || r;
  r = ((glMultiTexCoord4svARB = (PFNGLMULTITEXCOORD4SVARBPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4svARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_multitexture */

#ifdef GL_ARB_occlusion_query

static GLboolean _glewInit_GL_ARB_occlusion_query (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginQueryARB = (PFNGLBEGINQUERYARBPROC)glewGetProcAddress((const GLubyte*)"glBeginQueryARB")) == NULL) || r;
  r = ((glDeleteQueriesARB = (PFNGLDELETEQUERIESARBPROC)glewGetProcAddress((const GLubyte*)"glDeleteQueriesARB")) == NULL) || r;
  r = ((glEndQueryARB = (PFNGLENDQUERYARBPROC)glewGetProcAddress((const GLubyte*)"glEndQueryARB")) == NULL) || r;
  r = ((glGenQueriesARB = (PFNGLGENQUERIESARBPROC)glewGetProcAddress((const GLubyte*)"glGenQueriesARB")) == NULL) || r;
  r = ((glGetQueryObjectivARB = (PFNGLGETQUERYOBJECTIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjectivARB")) == NULL) || r;
  r = ((glGetQueryObjectuivARB = (PFNGLGETQUERYOBJECTUIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjectuivARB")) == NULL) || r;
  r = ((glGetQueryivARB = (PFNGLGETQUERYIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetQueryivARB")) == NULL) || r;
  r = ((glIsQueryARB = (PFNGLISQUERYARBPROC)glewGetProcAddress((const GLubyte*)"glIsQueryARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_occlusion_query */

#ifdef GL_ARB_parallel_shader_compile

static GLboolean _glewInit_GL_ARB_parallel_shader_compile (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMaxShaderCompilerThreadsARB = (PFNGLMAXSHADERCOMPILERTHREADSARBPROC)glewGetProcAddress((const GLubyte*)"glMaxShaderCompilerThreadsARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_parallel_shader_compile */

#ifdef GL_ARB_point_parameters

static GLboolean _glewInit_GL_ARB_point_parameters (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glPointParameterfARB = (PFNGLPOINTPARAMETERFARBPROC)glewGetProcAddress((const GLubyte*)"glPointParameterfARB")) == NULL) || r;
  r = ((glPointParameterfvARB = (PFNGLPOINTPARAMETERFVARBPROC)glewGetProcAddress((const GLubyte*)"glPointParameterfvARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_point_parameters */

#ifdef GL_ARB_program_interface_query

static GLboolean _glewInit_GL_ARB_program_interface_query (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetProgramInterfaceiv = (PFNGLGETPROGRAMINTERFACEIVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramInterfaceiv")) == NULL) || r;
  r = ((glGetProgramResourceIndex = (PFNGLGETPROGRAMRESOURCEINDEXPROC)glewGetProcAddress((const GLubyte*)"glGetProgramResourceIndex")) == NULL) || r;
  r = ((glGetProgramResourceLocation = (PFNGLGETPROGRAMRESOURCELOCATIONPROC)glewGetProcAddress((const GLubyte*)"glGetProgramResourceLocation")) == NULL) || r;
  r = ((glGetProgramResourceLocationIndex = (PFNGLGETPROGRAMRESOURCELOCATIONINDEXPROC)glewGetProcAddress((const GLubyte*)"glGetProgramResourceLocationIndex")) == NULL) || r;
  r = ((glGetProgramResourceName = (PFNGLGETPROGRAMRESOURCENAMEPROC)glewGetProcAddress((const GLubyte*)"glGetProgramResourceName")) == NULL) || r;
  r = ((glGetProgramResourceiv = (PFNGLGETPROGRAMRESOURCEIVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramResourceiv")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_program_interface_query */

#ifdef GL_ARB_provoking_vertex

static GLboolean _glewInit_GL_ARB_provoking_vertex (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glProvokingVertex = (PFNGLPROVOKINGVERTEXPROC)glewGetProcAddress((const GLubyte*)"glProvokingVertex")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_provoking_vertex */

#ifdef GL_ARB_robustness

static GLboolean _glewInit_GL_ARB_robustness (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetGraphicsResetStatusARB = (PFNGLGETGRAPHICSRESETSTATUSARBPROC)glewGetProcAddress((const GLubyte*)"glGetGraphicsResetStatusARB")) == NULL) || r;
  r = ((glGetnColorTableARB = (PFNGLGETNCOLORTABLEARBPROC)glewGetProcAddress((const GLubyte*)"glGetnColorTableARB")) == NULL) || r;
  r = ((glGetnCompressedTexImageARB = (PFNGLGETNCOMPRESSEDTEXIMAGEARBPROC)glewGetProcAddress((const GLubyte*)"glGetnCompressedTexImageARB")) == NULL) || r;
  r = ((glGetnConvolutionFilterARB = (PFNGLGETNCONVOLUTIONFILTERARBPROC)glewGetProcAddress((const GLubyte*)"glGetnConvolutionFilterARB")) == NULL) || r;
  r = ((glGetnHistogramARB = (PFNGLGETNHISTOGRAMARBPROC)glewGetProcAddress((const GLubyte*)"glGetnHistogramARB")) == NULL) || r;
  r = ((glGetnMapdvARB = (PFNGLGETNMAPDVARBPROC)glewGetProcAddress((const GLubyte*)"glGetnMapdvARB")) == NULL) || r;
  r = ((glGetnMapfvARB = (PFNGLGETNMAPFVARBPROC)glewGetProcAddress((const GLubyte*)"glGetnMapfvARB")) == NULL) || r;
  r = ((glGetnMapivARB = (PFNGLGETNMAPIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetnMapivARB")) == NULL) || r;
  r = ((glGetnMinmaxARB = (PFNGLGETNMINMAXARBPROC)glewGetProcAddress((const GLubyte*)"glGetnMinmaxARB")) == NULL) || r;
  r = ((glGetnPixelMapfvARB = (PFNGLGETNPIXELMAPFVARBPROC)glewGetProcAddress((const GLubyte*)"glGetnPixelMapfvARB")) == NULL) || r;
  r = ((glGetnPixelMapuivARB = (PFNGLGETNPIXELMAPUIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetnPixelMapuivARB")) == NULL) || r;
  r = ((glGetnPixelMapusvARB = (PFNGLGETNPIXELMAPUSVARBPROC)glewGetProcAddress((const GLubyte*)"glGetnPixelMapusvARB")) == NULL) || r;
  r = ((glGetnPolygonStippleARB = (PFNGLGETNPOLYGONSTIPPLEARBPROC)glewGetProcAddress((const GLubyte*)"glGetnPolygonStippleARB")) == NULL) || r;
  r = ((glGetnSeparableFilterARB = (PFNGLGETNSEPARABLEFILTERARBPROC)glewGetProcAddress((const GLubyte*)"glGetnSeparableFilterARB")) == NULL) || r;
  r = ((glGetnTexImageARB = (PFNGLGETNTEXIMAGEARBPROC)glewGetProcAddress((const GLubyte*)"glGetnTexImageARB")) == NULL) || r;
  r = ((glGetnUniformdvARB = (PFNGLGETNUNIFORMDVARBPROC)glewGetProcAddress((const GLubyte*)"glGetnUniformdvARB")) == NULL) || r;
  r = ((glGetnUniformfvARB = (PFNGLGETNUNIFORMFVARBPROC)glewGetProcAddress((const GLubyte*)"glGetnUniformfvARB")) == NULL) || r;
  r = ((glGetnUniformivARB = (PFNGLGETNUNIFORMIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetnUniformivARB")) == NULL) || r;
  r = ((glGetnUniformuivARB = (PFNGLGETNUNIFORMUIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetnUniformuivARB")) == NULL) || r;
  r = ((glReadnPixelsARB = (PFNGLREADNPIXELSARBPROC)glewGetProcAddress((const GLubyte*)"glReadnPixelsARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_robustness */

#ifdef GL_ARB_sample_locations

static GLboolean _glewInit_GL_ARB_sample_locations (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFramebufferSampleLocationsfvARB = (PFNGLFRAMEBUFFERSAMPLELOCATIONSFVARBPROC)glewGetProcAddress((const GLubyte*)"glFramebufferSampleLocationsfvARB")) == NULL) || r;
  r = ((glNamedFramebufferSampleLocationsfvARB = (PFNGLNAMEDFRAMEBUFFERSAMPLELOCATIONSFVARBPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferSampleLocationsfvARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_sample_locations */

#ifdef GL_ARB_sample_shading

static GLboolean _glewInit_GL_ARB_sample_shading (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMinSampleShadingARB = (PFNGLMINSAMPLESHADINGARBPROC)glewGetProcAddress((const GLubyte*)"glMinSampleShadingARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_sample_shading */

#ifdef GL_ARB_sampler_objects

static GLboolean _glewInit_GL_ARB_sampler_objects (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindSampler = (PFNGLBINDSAMPLERPROC)glewGetProcAddress((const GLubyte*)"glBindSampler")) == NULL) || r;
  r = ((glDeleteSamplers = (PFNGLDELETESAMPLERSPROC)glewGetProcAddress((const GLubyte*)"glDeleteSamplers")) == NULL) || r;
  r = ((glGenSamplers = (PFNGLGENSAMPLERSPROC)glewGetProcAddress((const GLubyte*)"glGenSamplers")) == NULL) || r;
  r = ((glGetSamplerParameterIiv = (PFNGLGETSAMPLERPARAMETERIIVPROC)glewGetProcAddress((const GLubyte*)"glGetSamplerParameterIiv")) == NULL) || r;
  r = ((glGetSamplerParameterIuiv = (PFNGLGETSAMPLERPARAMETERIUIVPROC)glewGetProcAddress((const GLubyte*)"glGetSamplerParameterIuiv")) == NULL) || r;
  r = ((glGetSamplerParameterfv = (PFNGLGETSAMPLERPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glGetSamplerParameterfv")) == NULL) || r;
  r = ((glGetSamplerParameteriv = (PFNGLGETSAMPLERPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glGetSamplerParameteriv")) == NULL) || r;
  r = ((glIsSampler = (PFNGLISSAMPLERPROC)glewGetProcAddress((const GLubyte*)"glIsSampler")) == NULL) || r;
  r = ((glSamplerParameterIiv = (PFNGLSAMPLERPARAMETERIIVPROC)glewGetProcAddress((const GLubyte*)"glSamplerParameterIiv")) == NULL) || r;
  r = ((glSamplerParameterIuiv = (PFNGLSAMPLERPARAMETERIUIVPROC)glewGetProcAddress((const GLubyte*)"glSamplerParameterIuiv")) == NULL) || r;
  r = ((glSamplerParameterf = (PFNGLSAMPLERPARAMETERFPROC)glewGetProcAddress((const GLubyte*)"glSamplerParameterf")) == NULL) || r;
  r = ((glSamplerParameterfv = (PFNGLSAMPLERPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glSamplerParameterfv")) == NULL) || r;
  r = ((glSamplerParameteri = (PFNGLSAMPLERPARAMETERIPROC)glewGetProcAddress((const GLubyte*)"glSamplerParameteri")) == NULL) || r;
  r = ((glSamplerParameteriv = (PFNGLSAMPLERPARAMETERIVPROC)glewGetProcAddress((const GLubyte*)"glSamplerParameteriv")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_sampler_objects */

#ifdef GL_ARB_separate_shader_objects

static GLboolean _glewInit_GL_ARB_separate_shader_objects (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glActiveShaderProgram = (PFNGLACTIVESHADERPROGRAMPROC)glewGetProcAddress((const GLubyte*)"glActiveShaderProgram")) == NULL) || r;
  r = ((glBindProgramPipeline = (PFNGLBINDPROGRAMPIPELINEPROC)glewGetProcAddress((const GLubyte*)"glBindProgramPipeline")) == NULL) || r;
  r = ((glCreateShaderProgramv = (PFNGLCREATESHADERPROGRAMVPROC)glewGetProcAddress((const GLubyte*)"glCreateShaderProgramv")) == NULL) || r;
  r = ((glDeleteProgramPipelines = (PFNGLDELETEPROGRAMPIPELINESPROC)glewGetProcAddress((const GLubyte*)"glDeleteProgramPipelines")) == NULL) || r;
  r = ((glGenProgramPipelines = (PFNGLGENPROGRAMPIPELINESPROC)glewGetProcAddress((const GLubyte*)"glGenProgramPipelines")) == NULL) || r;
  r = ((glGetProgramPipelineInfoLog = (PFNGLGETPROGRAMPIPELINEINFOLOGPROC)glewGetProcAddress((const GLubyte*)"glGetProgramPipelineInfoLog")) == NULL) || r;
  r = ((glGetProgramPipelineiv = (PFNGLGETPROGRAMPIPELINEIVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramPipelineiv")) == NULL) || r;
  r = ((glIsProgramPipeline = (PFNGLISPROGRAMPIPELINEPROC)glewGetProcAddress((const GLubyte*)"glIsProgramPipeline")) == NULL) || r;
  r = ((glProgramUniform1d = (PFNGLPROGRAMUNIFORM1DPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1d")) == NULL) || r;
  r = ((glProgramUniform1dv = (PFNGLPROGRAMUNIFORM1DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1dv")) == NULL) || r;
  r = ((glProgramUniform1f = (PFNGLPROGRAMUNIFORM1FPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1f")) == NULL) || r;
  r = ((glProgramUniform1fv = (PFNGLPROGRAMUNIFORM1FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1fv")) == NULL) || r;
  r = ((glProgramUniform1i = (PFNGLPROGRAMUNIFORM1IPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1i")) == NULL) || r;
  r = ((glProgramUniform1iv = (PFNGLPROGRAMUNIFORM1IVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1iv")) == NULL) || r;
  r = ((glProgramUniform1ui = (PFNGLPROGRAMUNIFORM1UIPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1ui")) == NULL) || r;
  r = ((glProgramUniform1uiv = (PFNGLPROGRAMUNIFORM1UIVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1uiv")) == NULL) || r;
  r = ((glProgramUniform2d = (PFNGLPROGRAMUNIFORM2DPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2d")) == NULL) || r;
  r = ((glProgramUniform2dv = (PFNGLPROGRAMUNIFORM2DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2dv")) == NULL) || r;
  r = ((glProgramUniform2f = (PFNGLPROGRAMUNIFORM2FPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2f")) == NULL) || r;
  r = ((glProgramUniform2fv = (PFNGLPROGRAMUNIFORM2FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2fv")) == NULL) || r;
  r = ((glProgramUniform2i = (PFNGLPROGRAMUNIFORM2IPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2i")) == NULL) || r;
  r = ((glProgramUniform2iv = (PFNGLPROGRAMUNIFORM2IVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2iv")) == NULL) || r;
  r = ((glProgramUniform2ui = (PFNGLPROGRAMUNIFORM2UIPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2ui")) == NULL) || r;
  r = ((glProgramUniform2uiv = (PFNGLPROGRAMUNIFORM2UIVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2uiv")) == NULL) || r;
  r = ((glProgramUniform3d = (PFNGLPROGRAMUNIFORM3DPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3d")) == NULL) || r;
  r = ((glProgramUniform3dv = (PFNGLPROGRAMUNIFORM3DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3dv")) == NULL) || r;
  r = ((glProgramUniform3f = (PFNGLPROGRAMUNIFORM3FPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3f")) == NULL) || r;
  r = ((glProgramUniform3fv = (PFNGLPROGRAMUNIFORM3FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3fv")) == NULL) || r;
  r = ((glProgramUniform3i = (PFNGLPROGRAMUNIFORM3IPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3i")) == NULL) || r;
  r = ((glProgramUniform3iv = (PFNGLPROGRAMUNIFORM3IVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3iv")) == NULL) || r;
  r = ((glProgramUniform3ui = (PFNGLPROGRAMUNIFORM3UIPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3ui")) == NULL) || r;
  r = ((glProgramUniform3uiv = (PFNGLPROGRAMUNIFORM3UIVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3uiv")) == NULL) || r;
  r = ((glProgramUniform4d = (PFNGLPROGRAMUNIFORM4DPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4d")) == NULL) || r;
  r = ((glProgramUniform4dv = (PFNGLPROGRAMUNIFORM4DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4dv")) == NULL) || r;
  r = ((glProgramUniform4f = (PFNGLPROGRAMUNIFORM4FPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4f")) == NULL) || r;
  r = ((glProgramUniform4fv = (PFNGLPROGRAMUNIFORM4FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4fv")) == NULL) || r;
  r = ((glProgramUniform4i = (PFNGLPROGRAMUNIFORM4IPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4i")) == NULL) || r;
  r = ((glProgramUniform4iv = (PFNGLPROGRAMUNIFORM4IVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4iv")) == NULL) || r;
  r = ((glProgramUniform4ui = (PFNGLPROGRAMUNIFORM4UIPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4ui")) == NULL) || r;
  r = ((glProgramUniform4uiv = (PFNGLPROGRAMUNIFORM4UIVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4uiv")) == NULL) || r;
  r = ((glProgramUniformMatrix2dv = (PFNGLPROGRAMUNIFORMMATRIX2DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix2dv")) == NULL) || r;
  r = ((glProgramUniformMatrix2fv = (PFNGLPROGRAMUNIFORMMATRIX2FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix2fv")) == NULL) || r;
  r = ((glProgramUniformMatrix2x3dv = (PFNGLPROGRAMUNIFORMMATRIX2X3DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix2x3dv")) == NULL) || r;
  r = ((glProgramUniformMatrix2x3fv = (PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix2x3fv")) == NULL) || r;
  r = ((glProgramUniformMatrix2x4dv = (PFNGLPROGRAMUNIFORMMATRIX2X4DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix2x4dv")) == NULL) || r;
  r = ((glProgramUniformMatrix2x4fv = (PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix2x4fv")) == NULL) || r;
  r = ((glProgramUniformMatrix3dv = (PFNGLPROGRAMUNIFORMMATRIX3DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix3dv")) == NULL) || r;
  r = ((glProgramUniformMatrix3fv = (PFNGLPROGRAMUNIFORMMATRIX3FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix3fv")) == NULL) || r;
  r = ((glProgramUniformMatrix3x2dv = (PFNGLPROGRAMUNIFORMMATRIX3X2DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix3x2dv")) == NULL) || r;
  r = ((glProgramUniformMatrix3x2fv = (PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix3x2fv")) == NULL) || r;
  r = ((glProgramUniformMatrix3x4dv = (PFNGLPROGRAMUNIFORMMATRIX3X4DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix3x4dv")) == NULL) || r;
  r = ((glProgramUniformMatrix3x4fv = (PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix3x4fv")) == NULL) || r;
  r = ((glProgramUniformMatrix4dv = (PFNGLPROGRAMUNIFORMMATRIX4DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix4dv")) == NULL) || r;
  r = ((glProgramUniformMatrix4fv = (PFNGLPROGRAMUNIFORMMATRIX4FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix4fv")) == NULL) || r;
  r = ((glProgramUniformMatrix4x2dv = (PFNGLPROGRAMUNIFORMMATRIX4X2DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix4x2dv")) == NULL) || r;
  r = ((glProgramUniformMatrix4x2fv = (PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix4x2fv")) == NULL) || r;
  r = ((glProgramUniformMatrix4x3dv = (PFNGLPROGRAMUNIFORMMATRIX4X3DVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix4x3dv")) == NULL) || r;
  r = ((glProgramUniformMatrix4x3fv = (PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix4x3fv")) == NULL) || r;
  r = ((glUseProgramStages = (PFNGLUSEPROGRAMSTAGESPROC)glewGetProcAddress((const GLubyte*)"glUseProgramStages")) == NULL) || r;
  r = ((glValidateProgramPipeline = (PFNGLVALIDATEPROGRAMPIPELINEPROC)glewGetProcAddress((const GLubyte*)"glValidateProgramPipeline")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_separate_shader_objects */

#ifdef GL_ARB_shader_atomic_counters

static GLboolean _glewInit_GL_ARB_shader_atomic_counters (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetActiveAtomicCounterBufferiv = (PFNGLGETACTIVEATOMICCOUNTERBUFFERIVPROC)glewGetProcAddress((const GLubyte*)"glGetActiveAtomicCounterBufferiv")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_shader_atomic_counters */

#ifdef GL_ARB_shader_image_load_store

static GLboolean _glewInit_GL_ARB_shader_image_load_store (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindImageTexture = (PFNGLBINDIMAGETEXTUREPROC)glewGetProcAddress((const GLubyte*)"glBindImageTexture")) == NULL) || r;
  r = ((glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)glewGetProcAddress((const GLubyte*)"glMemoryBarrier")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_shader_image_load_store */

#ifdef GL_ARB_shader_objects

static GLboolean _glewInit_GL_ARB_shader_objects (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glAttachObjectARB = (PFNGLATTACHOBJECTARBPROC)glewGetProcAddress((const GLubyte*)"glAttachObjectARB")) == NULL) || r;
  r = ((glCompileShaderARB = (PFNGLCOMPILESHADERARBPROC)glewGetProcAddress((const GLubyte*)"glCompileShaderARB")) == NULL) || r;
  r = ((glCreateProgramObjectARB = (PFNGLCREATEPROGRAMOBJECTARBPROC)glewGetProcAddress((const GLubyte*)"glCreateProgramObjectARB")) == NULL) || r;
  r = ((glCreateShaderObjectARB = (PFNGLCREATESHADEROBJECTARBPROC)glewGetProcAddress((const GLubyte*)"glCreateShaderObjectARB")) == NULL) || r;
  r = ((glDeleteObjectARB = (PFNGLDELETEOBJECTARBPROC)glewGetProcAddress((const GLubyte*)"glDeleteObjectARB")) == NULL) || r;
  r = ((glDetachObjectARB = (PFNGLDETACHOBJECTARBPROC)glewGetProcAddress((const GLubyte*)"glDetachObjectARB")) == NULL) || r;
  r = ((glGetActiveUniformARB = (PFNGLGETACTIVEUNIFORMARBPROC)glewGetProcAddress((const GLubyte*)"glGetActiveUniformARB")) == NULL) || r;
  r = ((glGetAttachedObjectsARB = (PFNGLGETATTACHEDOBJECTSARBPROC)glewGetProcAddress((const GLubyte*)"glGetAttachedObjectsARB")) == NULL) || r;
  r = ((glGetHandleARB = (PFNGLGETHANDLEARBPROC)glewGetProcAddress((const GLubyte*)"glGetHandleARB")) == NULL) || r;
  r = ((glGetInfoLogARB = (PFNGLGETINFOLOGARBPROC)glewGetProcAddress((const GLubyte*)"glGetInfoLogARB")) == NULL) || r;
  r = ((glGetObjectParameterfvARB = (PFNGLGETOBJECTPARAMETERFVARBPROC)glewGetProcAddress((const GLubyte*)"glGetObjectParameterfvARB")) == NULL) || r;
  r = ((glGetObjectParameterivARB = (PFNGLGETOBJECTPARAMETERIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetObjectParameterivARB")) == NULL) || r;
  r = ((glGetShaderSourceARB = (PFNGLGETSHADERSOURCEARBPROC)glewGetProcAddress((const GLubyte*)"glGetShaderSourceARB")) == NULL) || r;
  r = ((glGetUniformLocationARB = (PFNGLGETUNIFORMLOCATIONARBPROC)glewGetProcAddress((const GLubyte*)"glGetUniformLocationARB")) == NULL) || r;
  r = ((glGetUniformfvARB = (PFNGLGETUNIFORMFVARBPROC)glewGetProcAddress((const GLubyte*)"glGetUniformfvARB")) == NULL) || r;
  r = ((glGetUniformivARB = (PFNGLGETUNIFORMIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetUniformivARB")) == NULL) || r;
  r = ((glLinkProgramARB = (PFNGLLINKPROGRAMARBPROC)glewGetProcAddress((const GLubyte*)"glLinkProgramARB")) == NULL) || r;
  r = ((glShaderSourceARB = (PFNGLSHADERSOURCEARBPROC)glewGetProcAddress((const GLubyte*)"glShaderSourceARB")) == NULL) || r;
  r = ((glUniform1fARB = (PFNGLUNIFORM1FARBPROC)glewGetProcAddress((const GLubyte*)"glUniform1fARB")) == NULL) || r;
  r = ((glUniform1fvARB = (PFNGLUNIFORM1FVARBPROC)glewGetProcAddress((const GLubyte*)"glUniform1fvARB")) == NULL) || r;
  r = ((glUniform1iARB = (PFNGLUNIFORM1IARBPROC)glewGetProcAddress((const GLubyte*)"glUniform1iARB")) == NULL) || r;
  r = ((glUniform1ivARB = (PFNGLUNIFORM1IVARBPROC)glewGetProcAddress((const GLubyte*)"glUniform1ivARB")) == NULL) || r;
  r = ((glUniform2fARB = (PFNGLUNIFORM2FARBPROC)glewGetProcAddress((const GLubyte*)"glUniform2fARB")) == NULL) || r;
  r = ((glUniform2fvARB = (PFNGLUNIFORM2FVARBPROC)glewGetProcAddress((const GLubyte*)"glUniform2fvARB")) == NULL) || r;
  r = ((glUniform2iARB = (PFNGLUNIFORM2IARBPROC)glewGetProcAddress((const GLubyte*)"glUniform2iARB")) == NULL) || r;
  r = ((glUniform2ivARB = (PFNGLUNIFORM2IVARBPROC)glewGetProcAddress((const GLubyte*)"glUniform2ivARB")) == NULL) || r;
  r = ((glUniform3fARB = (PFNGLUNIFORM3FARBPROC)glewGetProcAddress((const GLubyte*)"glUniform3fARB")) == NULL) || r;
  r = ((glUniform3fvARB = (PFNGLUNIFORM3FVARBPROC)glewGetProcAddress((const GLubyte*)"glUniform3fvARB")) == NULL) || r;
  r = ((glUniform3iARB = (PFNGLUNIFORM3IARBPROC)glewGetProcAddress((const GLubyte*)"glUniform3iARB")) == NULL) || r;
  r = ((glUniform3ivARB = (PFNGLUNIFORM3IVARBPROC)glewGetProcAddress((const GLubyte*)"glUniform3ivARB")) == NULL) || r;
  r = ((glUniform4fARB = (PFNGLUNIFORM4FARBPROC)glewGetProcAddress((const GLubyte*)"glUniform4fARB")) == NULL) || r;
  r = ((glUniform4fvARB = (PFNGLUNIFORM4FVARBPROC)glewGetProcAddress((const GLubyte*)"glUniform4fvARB")) == NULL) || r;
  r = ((glUniform4iARB = (PFNGLUNIFORM4IARBPROC)glewGetProcAddress((const GLubyte*)"glUniform4iARB")) == NULL) || r;
  r = ((glUniform4ivARB = (PFNGLUNIFORM4IVARBPROC)glewGetProcAddress((const GLubyte*)"glUniform4ivARB")) == NULL) || r;
  r = ((glUniformMatrix2fvARB = (PFNGLUNIFORMMATRIX2FVARBPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix2fvARB")) == NULL) || r;
  r = ((glUniformMatrix3fvARB = (PFNGLUNIFORMMATRIX3FVARBPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix3fvARB")) == NULL) || r;
  r = ((glUniformMatrix4fvARB = (PFNGLUNIFORMMATRIX4FVARBPROC)glewGetProcAddress((const GLubyte*)"glUniformMatrix4fvARB")) == NULL) || r;
  r = ((glUseProgramObjectARB = (PFNGLUSEPROGRAMOBJECTARBPROC)glewGetProcAddress((const GLubyte*)"glUseProgramObjectARB")) == NULL) || r;
  r = ((glValidateProgramARB = (PFNGLVALIDATEPROGRAMARBPROC)glewGetProcAddress((const GLubyte*)"glValidateProgramARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_shader_objects */

#ifdef GL_ARB_shader_storage_buffer_object

static GLboolean _glewInit_GL_ARB_shader_storage_buffer_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glShaderStorageBlockBinding = (PFNGLSHADERSTORAGEBLOCKBINDINGPROC)glewGetProcAddress((const GLubyte*)"glShaderStorageBlockBinding")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_shader_storage_buffer_object */

#ifdef GL_ARB_shader_subroutine

static GLboolean _glewInit_GL_ARB_shader_subroutine (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetActiveSubroutineName = (PFNGLGETACTIVESUBROUTINENAMEPROC)glewGetProcAddress((const GLubyte*)"glGetActiveSubroutineName")) == NULL) || r;
  r = ((glGetActiveSubroutineUniformName = (PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC)glewGetProcAddress((const GLubyte*)"glGetActiveSubroutineUniformName")) == NULL) || r;
  r = ((glGetActiveSubroutineUniformiv = (PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC)glewGetProcAddress((const GLubyte*)"glGetActiveSubroutineUniformiv")) == NULL) || r;
  r = ((glGetProgramStageiv = (PFNGLGETPROGRAMSTAGEIVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramStageiv")) == NULL) || r;
  r = ((glGetSubroutineIndex = (PFNGLGETSUBROUTINEINDEXPROC)glewGetProcAddress((const GLubyte*)"glGetSubroutineIndex")) == NULL) || r;
  r = ((glGetSubroutineUniformLocation = (PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC)glewGetProcAddress((const GLubyte*)"glGetSubroutineUniformLocation")) == NULL) || r;
  r = ((glGetUniformSubroutineuiv = (PFNGLGETUNIFORMSUBROUTINEUIVPROC)glewGetProcAddress((const GLubyte*)"glGetUniformSubroutineuiv")) == NULL) || r;
  r = ((glUniformSubroutinesuiv = (PFNGLUNIFORMSUBROUTINESUIVPROC)glewGetProcAddress((const GLubyte*)"glUniformSubroutinesuiv")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_shader_subroutine */

#ifdef GL_ARB_shading_language_include

static GLboolean _glewInit_GL_ARB_shading_language_include (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCompileShaderIncludeARB = (PFNGLCOMPILESHADERINCLUDEARBPROC)glewGetProcAddress((const GLubyte*)"glCompileShaderIncludeARB")) == NULL) || r;
  r = ((glDeleteNamedStringARB = (PFNGLDELETENAMEDSTRINGARBPROC)glewGetProcAddress((const GLubyte*)"glDeleteNamedStringARB")) == NULL) || r;
  r = ((glGetNamedStringARB = (PFNGLGETNAMEDSTRINGARBPROC)glewGetProcAddress((const GLubyte*)"glGetNamedStringARB")) == NULL) || r;
  r = ((glGetNamedStringivARB = (PFNGLGETNAMEDSTRINGIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetNamedStringivARB")) == NULL) || r;
  r = ((glIsNamedStringARB = (PFNGLISNAMEDSTRINGARBPROC)glewGetProcAddress((const GLubyte*)"glIsNamedStringARB")) == NULL) || r;
  r = ((glNamedStringARB = (PFNGLNAMEDSTRINGARBPROC)glewGetProcAddress((const GLubyte*)"glNamedStringARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_shading_language_include */

#ifdef GL_ARB_sparse_buffer

static GLboolean _glewInit_GL_ARB_sparse_buffer (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBufferPageCommitmentARB = (PFNGLBUFFERPAGECOMMITMENTARBPROC)glewGetProcAddress((const GLubyte*)"glBufferPageCommitmentARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_sparse_buffer */

#ifdef GL_ARB_sparse_texture

static GLboolean _glewInit_GL_ARB_sparse_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexPageCommitmentARB = (PFNGLTEXPAGECOMMITMENTARBPROC)glewGetProcAddress((const GLubyte*)"glTexPageCommitmentARB")) == NULL) || r;
  r = ((glTexturePageCommitmentEXT = (PFNGLTEXTUREPAGECOMMITMENTEXTPROC)glewGetProcAddress((const GLubyte*)"glTexturePageCommitmentEXT")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_sparse_texture */

#ifdef GL_ARB_sync

static GLboolean _glewInit_GL_ARB_sync (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClientWaitSync = (PFNGLCLIENTWAITSYNCPROC)glewGetProcAddress((const GLubyte*)"glClientWaitSync")) == NULL) || r;
  r = ((glDeleteSync = (PFNGLDELETESYNCPROC)glewGetProcAddress((const GLubyte*)"glDeleteSync")) == NULL) || r;
  r = ((glFenceSync = (PFNGLFENCESYNCPROC)glewGetProcAddress((const GLubyte*)"glFenceSync")) == NULL) || r;
  r = ((glGetInteger64v = (PFNGLGETINTEGER64VPROC)glewGetProcAddress((const GLubyte*)"glGetInteger64v")) == NULL) || r;
  r = ((glGetSynciv = (PFNGLGETSYNCIVPROC)glewGetProcAddress((const GLubyte*)"glGetSynciv")) == NULL) || r;
  r = ((glIsSync = (PFNGLISSYNCPROC)glewGetProcAddress((const GLubyte*)"glIsSync")) == NULL) || r;
  r = ((glWaitSync = (PFNGLWAITSYNCPROC)glewGetProcAddress((const GLubyte*)"glWaitSync")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_sync */

#ifdef GL_ARB_tessellation_shader

static GLboolean _glewInit_GL_ARB_tessellation_shader (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glPatchParameterfv = (PFNGLPATCHPARAMETERFVPROC)glewGetProcAddress((const GLubyte*)"glPatchParameterfv")) == NULL) || r;
  r = ((glPatchParameteri = (PFNGLPATCHPARAMETERIPROC)glewGetProcAddress((const GLubyte*)"glPatchParameteri")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_tessellation_shader */

#ifdef GL_ARB_texture_barrier

static GLboolean _glewInit_GL_ARB_texture_barrier (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTextureBarrier = (PFNGLTEXTUREBARRIERPROC)glewGetProcAddress((const GLubyte*)"glTextureBarrier")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_texture_barrier */

#ifdef GL_ARB_texture_buffer_object

static GLboolean _glewInit_GL_ARB_texture_buffer_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexBufferARB = (PFNGLTEXBUFFERARBPROC)glewGetProcAddress((const GLubyte*)"glTexBufferARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_texture_buffer_object */

#ifdef GL_ARB_texture_buffer_range

static GLboolean _glewInit_GL_ARB_texture_buffer_range (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexBufferRange = (PFNGLTEXBUFFERRANGEPROC)glewGetProcAddress((const GLubyte*)"glTexBufferRange")) == NULL) || r;
  r = ((glTextureBufferRangeEXT = (PFNGLTEXTUREBUFFERRANGEEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureBufferRangeEXT")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_texture_buffer_range */

#ifdef GL_ARB_texture_compression

static GLboolean _glewInit_GL_ARB_texture_compression (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCompressedTexImage1DARB = (PFNGLCOMPRESSEDTEXIMAGE1DARBPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexImage1DARB")) == NULL) || r;
  r = ((glCompressedTexImage2DARB = (PFNGLCOMPRESSEDTEXIMAGE2DARBPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexImage2DARB")) == NULL) || r;
  r = ((glCompressedTexImage3DARB = (PFNGLCOMPRESSEDTEXIMAGE3DARBPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexImage3DARB")) == NULL) || r;
  r = ((glCompressedTexSubImage1DARB = (PFNGLCOMPRESSEDTEXSUBIMAGE1DARBPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexSubImage1DARB")) == NULL) || r;
  r = ((glCompressedTexSubImage2DARB = (PFNGLCOMPRESSEDTEXSUBIMAGE2DARBPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexSubImage2DARB")) == NULL) || r;
  r = ((glCompressedTexSubImage3DARB = (PFNGLCOMPRESSEDTEXSUBIMAGE3DARBPROC)glewGetProcAddress((const GLubyte*)"glCompressedTexSubImage3DARB")) == NULL) || r;
  r = ((glGetCompressedTexImageARB = (PFNGLGETCOMPRESSEDTEXIMAGEARBPROC)glewGetProcAddress((const GLubyte*)"glGetCompressedTexImageARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_texture_compression */

#ifdef GL_ARB_texture_multisample

static GLboolean _glewInit_GL_ARB_texture_multisample (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetMultisamplefv = (PFNGLGETMULTISAMPLEFVPROC)glewGetProcAddress((const GLubyte*)"glGetMultisamplefv")) == NULL) || r;
  r = ((glSampleMaski = (PFNGLSAMPLEMASKIPROC)glewGetProcAddress((const GLubyte*)"glSampleMaski")) == NULL) || r;
  r = ((glTexImage2DMultisample = (PFNGLTEXIMAGE2DMULTISAMPLEPROC)glewGetProcAddress((const GLubyte*)"glTexImage2DMultisample")) == NULL) || r;
  r = ((glTexImage3DMultisample = (PFNGLTEXIMAGE3DMULTISAMPLEPROC)glewGetProcAddress((const GLubyte*)"glTexImage3DMultisample")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_texture_multisample */

#ifdef GL_ARB_texture_storage

static GLboolean _glewInit_GL_ARB_texture_storage (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexStorage1D = (PFNGLTEXSTORAGE1DPROC)glewGetProcAddress((const GLubyte*)"glTexStorage1D")) == NULL) || r;
  r = ((glTexStorage2D = (PFNGLTEXSTORAGE2DPROC)glewGetProcAddress((const GLubyte*)"glTexStorage2D")) == NULL) || r;
  r = ((glTexStorage3D = (PFNGLTEXSTORAGE3DPROC)glewGetProcAddress((const GLubyte*)"glTexStorage3D")) == NULL) || r;
  r = ((glTextureStorage1DEXT = (PFNGLTEXTURESTORAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureStorage1DEXT")) == NULL) || r;
  r = ((glTextureStorage2DEXT = (PFNGLTEXTURESTORAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureStorage2DEXT")) == NULL) || r;
  r = ((glTextureStorage3DEXT = (PFNGLTEXTURESTORAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureStorage3DEXT")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_texture_storage */

#ifdef GL_ARB_texture_storage_multisample

static GLboolean _glewInit_GL_ARB_texture_storage_multisample (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexStorage2DMultisample = (PFNGLTEXSTORAGE2DMULTISAMPLEPROC)glewGetProcAddress((const GLubyte*)"glTexStorage2DMultisample")) == NULL) || r;
  r = ((glTexStorage3DMultisample = (PFNGLTEXSTORAGE3DMULTISAMPLEPROC)glewGetProcAddress((const GLubyte*)"glTexStorage3DMultisample")) == NULL) || r;
  r = ((glTextureStorage2DMultisampleEXT = (PFNGLTEXTURESTORAGE2DMULTISAMPLEEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureStorage2DMultisampleEXT")) == NULL) || r;
  r = ((glTextureStorage3DMultisampleEXT = (PFNGLTEXTURESTORAGE3DMULTISAMPLEEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureStorage3DMultisampleEXT")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_texture_storage_multisample */

#ifdef GL_ARB_texture_view

static GLboolean _glewInit_GL_ARB_texture_view (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTextureView = (PFNGLTEXTUREVIEWPROC)glewGetProcAddress((const GLubyte*)"glTextureView")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_texture_view */

#ifdef GL_ARB_timer_query

static GLboolean _glewInit_GL_ARB_timer_query (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetQueryObjecti64v = (PFNGLGETQUERYOBJECTI64VPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjecti64v")) == NULL) || r;
  r = ((glGetQueryObjectui64v = (PFNGLGETQUERYOBJECTUI64VPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjectui64v")) == NULL) || r;
  r = ((glQueryCounter = (PFNGLQUERYCOUNTERPROC)glewGetProcAddress((const GLubyte*)"glQueryCounter")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_timer_query */

#ifdef GL_ARB_transform_feedback2

static GLboolean _glewInit_GL_ARB_transform_feedback2 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindTransformFeedback = (PFNGLBINDTRANSFORMFEEDBACKPROC)glewGetProcAddress((const GLubyte*)"glBindTransformFeedback")) == NULL) || r;
  r = ((glDeleteTransformFeedbacks = (PFNGLDELETETRANSFORMFEEDBACKSPROC)glewGetProcAddress((const GLubyte*)"glDeleteTransformFeedbacks")) == NULL) || r;
  r = ((glDrawTransformFeedback = (PFNGLDRAWTRANSFORMFEEDBACKPROC)glewGetProcAddress((const GLubyte*)"glDrawTransformFeedback")) == NULL) || r;
  r = ((glGenTransformFeedbacks = (PFNGLGENTRANSFORMFEEDBACKSPROC)glewGetProcAddress((const GLubyte*)"glGenTransformFeedbacks")) == NULL) || r;
  r = ((glIsTransformFeedback = (PFNGLISTRANSFORMFEEDBACKPROC)glewGetProcAddress((const GLubyte*)"glIsTransformFeedback")) == NULL) || r;
  r = ((glPauseTransformFeedback = (PFNGLPAUSETRANSFORMFEEDBACKPROC)glewGetProcAddress((const GLubyte*)"glPauseTransformFeedback")) == NULL) || r;
  r = ((glResumeTransformFeedback = (PFNGLRESUMETRANSFORMFEEDBACKPROC)glewGetProcAddress((const GLubyte*)"glResumeTransformFeedback")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_transform_feedback2 */

#ifdef GL_ARB_transform_feedback3

static GLboolean _glewInit_GL_ARB_transform_feedback3 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginQueryIndexed = (PFNGLBEGINQUERYINDEXEDPROC)glewGetProcAddress((const GLubyte*)"glBeginQueryIndexed")) == NULL) || r;
  r = ((glDrawTransformFeedbackStream = (PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC)glewGetProcAddress((const GLubyte*)"glDrawTransformFeedbackStream")) == NULL) || r;
  r = ((glEndQueryIndexed = (PFNGLENDQUERYINDEXEDPROC)glewGetProcAddress((const GLubyte*)"glEndQueryIndexed")) == NULL) || r;
  r = ((glGetQueryIndexediv = (PFNGLGETQUERYINDEXEDIVPROC)glewGetProcAddress((const GLubyte*)"glGetQueryIndexediv")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_transform_feedback3 */

#ifdef GL_ARB_transform_feedback_instanced

static GLboolean _glewInit_GL_ARB_transform_feedback_instanced (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawTransformFeedbackInstanced = (PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC)glewGetProcAddress((const GLubyte*)"glDrawTransformFeedbackInstanced")) == NULL) || r;
  r = ((glDrawTransformFeedbackStreamInstanced = (PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC)glewGetProcAddress((const GLubyte*)"glDrawTransformFeedbackStreamInstanced")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_transform_feedback_instanced */

#ifdef GL_ARB_transpose_matrix

static GLboolean _glewInit_GL_ARB_transpose_matrix (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glLoadTransposeMatrixdARB = (PFNGLLOADTRANSPOSEMATRIXDARBPROC)glewGetProcAddress((const GLubyte*)"glLoadTransposeMatrixdARB")) == NULL) || r;
  r = ((glLoadTransposeMatrixfARB = (PFNGLLOADTRANSPOSEMATRIXFARBPROC)glewGetProcAddress((const GLubyte*)"glLoadTransposeMatrixfARB")) == NULL) || r;
  r = ((glMultTransposeMatrixdARB = (PFNGLMULTTRANSPOSEMATRIXDARBPROC)glewGetProcAddress((const GLubyte*)"glMultTransposeMatrixdARB")) == NULL) || r;
  r = ((glMultTransposeMatrixfARB = (PFNGLMULTTRANSPOSEMATRIXFARBPROC)glewGetProcAddress((const GLubyte*)"glMultTransposeMatrixfARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_transpose_matrix */

#ifdef GL_ARB_uniform_buffer_object

static GLboolean _glewInit_GL_ARB_uniform_buffer_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindBufferBase = (PFNGLBINDBUFFERBASEPROC)glewGetProcAddress((const GLubyte*)"glBindBufferBase")) == NULL) || r;
  r = ((glBindBufferRange = (PFNGLBINDBUFFERRANGEPROC)glewGetProcAddress((const GLubyte*)"glBindBufferRange")) == NULL) || r;
  r = ((glGetActiveUniformBlockName = (PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC)glewGetProcAddress((const GLubyte*)"glGetActiveUniformBlockName")) == NULL) || r;
  r = ((glGetActiveUniformBlockiv = (PFNGLGETACTIVEUNIFORMBLOCKIVPROC)glewGetProcAddress((const GLubyte*)"glGetActiveUniformBlockiv")) == NULL) || r;
  r = ((glGetActiveUniformName = (PFNGLGETACTIVEUNIFORMNAMEPROC)glewGetProcAddress((const GLubyte*)"glGetActiveUniformName")) == NULL) || r;
  r = ((glGetActiveUniformsiv = (PFNGLGETACTIVEUNIFORMSIVPROC)glewGetProcAddress((const GLubyte*)"glGetActiveUniformsiv")) == NULL) || r;
  r = ((glGetIntegeri_v = (PFNGLGETINTEGERI_VPROC)glewGetProcAddress((const GLubyte*)"glGetIntegeri_v")) == NULL) || r;
  r = ((glGetUniformBlockIndex = (PFNGLGETUNIFORMBLOCKINDEXPROC)glewGetProcAddress((const GLubyte*)"glGetUniformBlockIndex")) == NULL) || r;
  r = ((glGetUniformIndices = (PFNGLGETUNIFORMINDICESPROC)glewGetProcAddress((const GLubyte*)"glGetUniformIndices")) == NULL) || r;
  r = ((glUniformBlockBinding = (PFNGLUNIFORMBLOCKBINDINGPROC)glewGetProcAddress((const GLubyte*)"glUniformBlockBinding")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_uniform_buffer_object */

#ifdef GL_ARB_vertex_array_object

static GLboolean _glewInit_GL_ARB_vertex_array_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)glewGetProcAddress((const GLubyte*)"glBindVertexArray")) == NULL) || r;
  r = ((glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)glewGetProcAddress((const GLubyte*)"glDeleteVertexArrays")) == NULL) || r;
  r = ((glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)glewGetProcAddress((const GLubyte*)"glGenVertexArrays")) == NULL) || r;
  r = ((glIsVertexArray = (PFNGLISVERTEXARRAYPROC)glewGetProcAddress((const GLubyte*)"glIsVertexArray")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_vertex_array_object */

#ifdef GL_ARB_vertex_attrib_64bit

static GLboolean _glewInit_GL_ARB_vertex_attrib_64bit (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetVertexAttribLdv = (PFNGLGETVERTEXATTRIBLDVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribLdv")) == NULL) || r;
  r = ((glVertexAttribL1d = (PFNGLVERTEXATTRIBL1DPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL1d")) == NULL) || r;
  r = ((glVertexAttribL1dv = (PFNGLVERTEXATTRIBL1DVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL1dv")) == NULL) || r;
  r = ((glVertexAttribL2d = (PFNGLVERTEXATTRIBL2DPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL2d")) == NULL) || r;
  r = ((glVertexAttribL2dv = (PFNGLVERTEXATTRIBL2DVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL2dv")) == NULL) || r;
  r = ((glVertexAttribL3d = (PFNGLVERTEXATTRIBL3DPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL3d")) == NULL) || r;
  r = ((glVertexAttribL3dv = (PFNGLVERTEXATTRIBL3DVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL3dv")) == NULL) || r;
  r = ((glVertexAttribL4d = (PFNGLVERTEXATTRIBL4DPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL4d")) == NULL) || r;
  r = ((glVertexAttribL4dv = (PFNGLVERTEXATTRIBL4DVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL4dv")) == NULL) || r;
  r = ((glVertexAttribLPointer = (PFNGLVERTEXATTRIBLPOINTERPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribLPointer")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_vertex_attrib_64bit */

#ifdef GL_ARB_vertex_attrib_binding

static GLboolean _glewInit_GL_ARB_vertex_attrib_binding (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindVertexBuffer = (PFNGLBINDVERTEXBUFFERPROC)glewGetProcAddress((const GLubyte*)"glBindVertexBuffer")) == NULL) || r;
  r = ((glVertexArrayBindVertexBufferEXT = (PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayBindVertexBufferEXT")) == NULL) || r;
  r = ((glVertexArrayVertexAttribBindingEXT = (PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexAttribBindingEXT")) == NULL) || r;
  r = ((glVertexArrayVertexAttribFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexAttribFormatEXT")) == NULL) || r;
  r = ((glVertexArrayVertexAttribIFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexAttribIFormatEXT")) == NULL) || r;
  r = ((glVertexArrayVertexAttribLFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexAttribLFormatEXT")) == NULL) || r;
  r = ((glVertexArrayVertexBindingDivisorEXT = (PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexBindingDivisorEXT")) == NULL) || r;
  r = ((glVertexAttribBinding = (PFNGLVERTEXATTRIBBINDINGPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribBinding")) == NULL) || r;
  r = ((glVertexAttribFormat = (PFNGLVERTEXATTRIBFORMATPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribFormat")) == NULL) || r;
  r = ((glVertexAttribIFormat = (PFNGLVERTEXATTRIBIFORMATPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribIFormat")) == NULL) || r;
  r = ((glVertexAttribLFormat = (PFNGLVERTEXATTRIBLFORMATPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribLFormat")) == NULL) || r;
  r = ((glVertexBindingDivisor = (PFNGLVERTEXBINDINGDIVISORPROC)glewGetProcAddress((const GLubyte*)"glVertexBindingDivisor")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_vertex_attrib_binding */

#ifdef GL_ARB_vertex_blend

static GLboolean _glewInit_GL_ARB_vertex_blend (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glVertexBlendARB = (PFNGLVERTEXBLENDARBPROC)glewGetProcAddress((const GLubyte*)"glVertexBlendARB")) == NULL) || r;
  r = ((glWeightPointerARB = (PFNGLWEIGHTPOINTERARBPROC)glewGetProcAddress((const GLubyte*)"glWeightPointerARB")) == NULL) || r;
  r = ((glWeightbvARB = (PFNGLWEIGHTBVARBPROC)glewGetProcAddress((const GLubyte*)"glWeightbvARB")) == NULL) || r;
  r = ((glWeightdvARB = (PFNGLWEIGHTDVARBPROC)glewGetProcAddress((const GLubyte*)"glWeightdvARB")) == NULL) || r;
  r = ((glWeightfvARB = (PFNGLWEIGHTFVARBPROC)glewGetProcAddress((const GLubyte*)"glWeightfvARB")) == NULL) || r;
  r = ((glWeightivARB = (PFNGLWEIGHTIVARBPROC)glewGetProcAddress((const GLubyte*)"glWeightivARB")) == NULL) || r;
  r = ((glWeightsvARB = (PFNGLWEIGHTSVARBPROC)glewGetProcAddress((const GLubyte*)"glWeightsvARB")) == NULL) || r;
  r = ((glWeightubvARB = (PFNGLWEIGHTUBVARBPROC)glewGetProcAddress((const GLubyte*)"glWeightubvARB")) == NULL) || r;
  r = ((glWeightuivARB = (PFNGLWEIGHTUIVARBPROC)glewGetProcAddress((const GLubyte*)"glWeightuivARB")) == NULL) || r;
  r = ((glWeightusvARB = (PFNGLWEIGHTUSVARBPROC)glewGetProcAddress((const GLubyte*)"glWeightusvARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_vertex_blend */

#ifdef GL_ARB_vertex_buffer_object

static GLboolean _glewInit_GL_ARB_vertex_buffer_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindBufferARB = (PFNGLBINDBUFFERARBPROC)glewGetProcAddress((const GLubyte*)"glBindBufferARB")) == NULL) || r;
  r = ((glBufferDataARB = (PFNGLBUFFERDATAARBPROC)glewGetProcAddress((const GLubyte*)"glBufferDataARB")) == NULL) || r;
  r = ((glBufferSubDataARB = (PFNGLBUFFERSUBDATAARBPROC)glewGetProcAddress((const GLubyte*)"glBufferSubDataARB")) == NULL) || r;
  r = ((glDeleteBuffersARB = (PFNGLDELETEBUFFERSARBPROC)glewGetProcAddress((const GLubyte*)"glDeleteBuffersARB")) == NULL) || r;
  r = ((glGenBuffersARB = (PFNGLGENBUFFERSARBPROC)glewGetProcAddress((const GLubyte*)"glGenBuffersARB")) == NULL) || r;
  r = ((glGetBufferParameterivARB = (PFNGLGETBUFFERPARAMETERIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetBufferParameterivARB")) == NULL) || r;
  r = ((glGetBufferPointervARB = (PFNGLGETBUFFERPOINTERVARBPROC)glewGetProcAddress((const GLubyte*)"glGetBufferPointervARB")) == NULL) || r;
  r = ((glGetBufferSubDataARB = (PFNGLGETBUFFERSUBDATAARBPROC)glewGetProcAddress((const GLubyte*)"glGetBufferSubDataARB")) == NULL) || r;
  r = ((glIsBufferARB = (PFNGLISBUFFERARBPROC)glewGetProcAddress((const GLubyte*)"glIsBufferARB")) == NULL) || r;
  r = ((glMapBufferARB = (PFNGLMAPBUFFERARBPROC)glewGetProcAddress((const GLubyte*)"glMapBufferARB")) == NULL) || r;
  r = ((glUnmapBufferARB = (PFNGLUNMAPBUFFERARBPROC)glewGetProcAddress((const GLubyte*)"glUnmapBufferARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_vertex_buffer_object */

#ifdef GL_ARB_vertex_program

static GLboolean _glewInit_GL_ARB_vertex_program (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindProgramARB = (PFNGLBINDPROGRAMARBPROC)glewGetProcAddress((const GLubyte*)"glBindProgramARB")) == NULL) || r;
  r = ((glDeleteProgramsARB = (PFNGLDELETEPROGRAMSARBPROC)glewGetProcAddress((const GLubyte*)"glDeleteProgramsARB")) == NULL) || r;
  r = ((glDisableVertexAttribArrayARB = (PFNGLDISABLEVERTEXATTRIBARRAYARBPROC)glewGetProcAddress((const GLubyte*)"glDisableVertexAttribArrayARB")) == NULL) || r;
  r = ((glEnableVertexAttribArrayARB = (PFNGLENABLEVERTEXATTRIBARRAYARBPROC)glewGetProcAddress((const GLubyte*)"glEnableVertexAttribArrayARB")) == NULL) || r;
  r = ((glGenProgramsARB = (PFNGLGENPROGRAMSARBPROC)glewGetProcAddress((const GLubyte*)"glGenProgramsARB")) == NULL) || r;
  r = ((glGetProgramEnvParameterdvARB = (PFNGLGETPROGRAMENVPARAMETERDVARBPROC)glewGetProcAddress((const GLubyte*)"glGetProgramEnvParameterdvARB")) == NULL) || r;
  r = ((glGetProgramEnvParameterfvARB = (PFNGLGETPROGRAMENVPARAMETERFVARBPROC)glewGetProcAddress((const GLubyte*)"glGetProgramEnvParameterfvARB")) == NULL) || r;
  r = ((glGetProgramLocalParameterdvARB = (PFNGLGETPROGRAMLOCALPARAMETERDVARBPROC)glewGetProcAddress((const GLubyte*)"glGetProgramLocalParameterdvARB")) == NULL) || r;
  r = ((glGetProgramLocalParameterfvARB = (PFNGLGETPROGRAMLOCALPARAMETERFVARBPROC)glewGetProcAddress((const GLubyte*)"glGetProgramLocalParameterfvARB")) == NULL) || r;
  r = ((glGetProgramStringARB = (PFNGLGETPROGRAMSTRINGARBPROC)glewGetProcAddress((const GLubyte*)"glGetProgramStringARB")) == NULL) || r;
  r = ((glGetProgramivARB = (PFNGLGETPROGRAMIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetProgramivARB")) == NULL) || r;
  r = ((glGetVertexAttribPointervARB = (PFNGLGETVERTEXATTRIBPOINTERVARBPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribPointervARB")) == NULL) || r;
  r = ((glGetVertexAttribdvARB = (PFNGLGETVERTEXATTRIBDVARBPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribdvARB")) == NULL) || r;
  r = ((glGetVertexAttribfvARB = (PFNGLGETVERTEXATTRIBFVARBPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribfvARB")) == NULL) || r;
  r = ((glGetVertexAttribivARB = (PFNGLGETVERTEXATTRIBIVARBPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribivARB")) == NULL) || r;
  r = ((glIsProgramARB = (PFNGLISPROGRAMARBPROC)glewGetProcAddress((const GLubyte*)"glIsProgramARB")) == NULL) || r;
  r = ((glProgramEnvParameter4dARB = (PFNGLPROGRAMENVPARAMETER4DARBPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParameter4dARB")) == NULL) || r;
  r = ((glProgramEnvParameter4dvARB = (PFNGLPROGRAMENVPARAMETER4DVARBPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParameter4dvARB")) == NULL) || r;
  r = ((glProgramEnvParameter4fARB = (PFNGLPROGRAMENVPARAMETER4FARBPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParameter4fARB")) == NULL) || r;
  r = ((glProgramEnvParameter4fvARB = (PFNGLPROGRAMENVPARAMETER4FVARBPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParameter4fvARB")) == NULL) || r;
  r = ((glProgramLocalParameter4dARB = (PFNGLPROGRAMLOCALPARAMETER4DARBPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParameter4dARB")) == NULL) || r;
  r = ((glProgramLocalParameter4dvARB = (PFNGLPROGRAMLOCALPARAMETER4DVARBPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParameter4dvARB")) == NULL) || r;
  r = ((glProgramLocalParameter4fARB = (PFNGLPROGRAMLOCALPARAMETER4FARBPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParameter4fARB")) == NULL) || r;
  r = ((glProgramLocalParameter4fvARB = (PFNGLPROGRAMLOCALPARAMETER4FVARBPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParameter4fvARB")) == NULL) || r;
  r = ((glProgramStringARB = (PFNGLPROGRAMSTRINGARBPROC)glewGetProcAddress((const GLubyte*)"glProgramStringARB")) == NULL) || r;
  r = ((glVertexAttrib1dARB = (PFNGLVERTEXATTRIB1DARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1dARB")) == NULL) || r;
  r = ((glVertexAttrib1dvARB = (PFNGLVERTEXATTRIB1DVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1dvARB")) == NULL) || r;
  r = ((glVertexAttrib1fARB = (PFNGLVERTEXATTRIB1FARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1fARB")) == NULL) || r;
  r = ((glVertexAttrib1fvARB = (PFNGLVERTEXATTRIB1FVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1fvARB")) == NULL) || r;
  r = ((glVertexAttrib1sARB = (PFNGLVERTEXATTRIB1SARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1sARB")) == NULL) || r;
  r = ((glVertexAttrib1svARB = (PFNGLVERTEXATTRIB1SVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1svARB")) == NULL) || r;
  r = ((glVertexAttrib2dARB = (PFNGLVERTEXATTRIB2DARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2dARB")) == NULL) || r;
  r = ((glVertexAttrib2dvARB = (PFNGLVERTEXATTRIB2DVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2dvARB")) == NULL) || r;
  r = ((glVertexAttrib2fARB = (PFNGLVERTEXATTRIB2FARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2fARB")) == NULL) || r;
  r = ((glVertexAttrib2fvARB = (PFNGLVERTEXATTRIB2FVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2fvARB")) == NULL) || r;
  r = ((glVertexAttrib2sARB = (PFNGLVERTEXATTRIB2SARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2sARB")) == NULL) || r;
  r = ((glVertexAttrib2svARB = (PFNGLVERTEXATTRIB2SVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2svARB")) == NULL) || r;
  r = ((glVertexAttrib3dARB = (PFNGLVERTEXATTRIB3DARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3dARB")) == NULL) || r;
  r = ((glVertexAttrib3dvARB = (PFNGLVERTEXATTRIB3DVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3dvARB")) == NULL) || r;
  r = ((glVertexAttrib3fARB = (PFNGLVERTEXATTRIB3FARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3fARB")) == NULL) || r;
  r = ((glVertexAttrib3fvARB = (PFNGLVERTEXATTRIB3FVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3fvARB")) == NULL) || r;
  r = ((glVertexAttrib3sARB = (PFNGLVERTEXATTRIB3SARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3sARB")) == NULL) || r;
  r = ((glVertexAttrib3svARB = (PFNGLVERTEXATTRIB3SVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3svARB")) == NULL) || r;
  r = ((glVertexAttrib4NbvARB = (PFNGLVERTEXATTRIB4NBVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4NbvARB")) == NULL) || r;
  r = ((glVertexAttrib4NivARB = (PFNGLVERTEXATTRIB4NIVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4NivARB")) == NULL) || r;
  r = ((glVertexAttrib4NsvARB = (PFNGLVERTEXATTRIB4NSVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4NsvARB")) == NULL) || r;
  r = ((glVertexAttrib4NubARB = (PFNGLVERTEXATTRIB4NUBARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4NubARB")) == NULL) || r;
  r = ((glVertexAttrib4NubvARB = (PFNGLVERTEXATTRIB4NUBVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4NubvARB")) == NULL) || r;
  r = ((glVertexAttrib4NuivARB = (PFNGLVERTEXATTRIB4NUIVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4NuivARB")) == NULL) || r;
  r = ((glVertexAttrib4NusvARB = (PFNGLVERTEXATTRIB4NUSVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4NusvARB")) == NULL) || r;
  r = ((glVertexAttrib4bvARB = (PFNGLVERTEXATTRIB4BVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4bvARB")) == NULL) || r;
  r = ((glVertexAttrib4dARB = (PFNGLVERTEXATTRIB4DARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4dARB")) == NULL) || r;
  r = ((glVertexAttrib4dvARB = (PFNGLVERTEXATTRIB4DVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4dvARB")) == NULL) || r;
  r = ((glVertexAttrib4fARB = (PFNGLVERTEXATTRIB4FARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4fARB")) == NULL) || r;
  r = ((glVertexAttrib4fvARB = (PFNGLVERTEXATTRIB4FVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4fvARB")) == NULL) || r;
  r = ((glVertexAttrib4ivARB = (PFNGLVERTEXATTRIB4IVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4ivARB")) == NULL) || r;
  r = ((glVertexAttrib4sARB = (PFNGLVERTEXATTRIB4SARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4sARB")) == NULL) || r;
  r = ((glVertexAttrib4svARB = (PFNGLVERTEXATTRIB4SVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4svARB")) == NULL) || r;
  r = ((glVertexAttrib4ubvARB = (PFNGLVERTEXATTRIB4UBVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4ubvARB")) == NULL) || r;
  r = ((glVertexAttrib4uivARB = (PFNGLVERTEXATTRIB4UIVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4uivARB")) == NULL) || r;
  r = ((glVertexAttrib4usvARB = (PFNGLVERTEXATTRIB4USVARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4usvARB")) == NULL) || r;
  r = ((glVertexAttribPointerARB = (PFNGLVERTEXATTRIBPOINTERARBPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribPointerARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_vertex_program */

#ifdef GL_ARB_vertex_shader

static GLboolean _glewInit_GL_ARB_vertex_shader (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindAttribLocationARB = (PFNGLBINDATTRIBLOCATIONARBPROC)glewGetProcAddress((const GLubyte*)"glBindAttribLocationARB")) == NULL) || r;
  r = ((glGetActiveAttribARB = (PFNGLGETACTIVEATTRIBARBPROC)glewGetProcAddress((const GLubyte*)"glGetActiveAttribARB")) == NULL) || r;
  r = ((glGetAttribLocationARB = (PFNGLGETATTRIBLOCATIONARBPROC)glewGetProcAddress((const GLubyte*)"glGetAttribLocationARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_vertex_shader */

#ifdef GL_ARB_vertex_type_2_10_10_10_rev

static GLboolean _glewInit_GL_ARB_vertex_type_2_10_10_10_rev (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glColorP3ui = (PFNGLCOLORP3UIPROC)glewGetProcAddress((const GLubyte*)"glColorP3ui")) == NULL) || r;
  r = ((glColorP3uiv = (PFNGLCOLORP3UIVPROC)glewGetProcAddress((const GLubyte*)"glColorP3uiv")) == NULL) || r;
  r = ((glColorP4ui = (PFNGLCOLORP4UIPROC)glewGetProcAddress((const GLubyte*)"glColorP4ui")) == NULL) || r;
  r = ((glColorP4uiv = (PFNGLCOLORP4UIVPROC)glewGetProcAddress((const GLubyte*)"glColorP4uiv")) == NULL) || r;
  r = ((glMultiTexCoordP1ui = (PFNGLMULTITEXCOORDP1UIPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoordP1ui")) == NULL) || r;
  r = ((glMultiTexCoordP1uiv = (PFNGLMULTITEXCOORDP1UIVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoordP1uiv")) == NULL) || r;
  r = ((glMultiTexCoordP2ui = (PFNGLMULTITEXCOORDP2UIPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoordP2ui")) == NULL) || r;
  r = ((glMultiTexCoordP2uiv = (PFNGLMULTITEXCOORDP2UIVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoordP2uiv")) == NULL) || r;
  r = ((glMultiTexCoordP3ui = (PFNGLMULTITEXCOORDP3UIPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoordP3ui")) == NULL) || r;
  r = ((glMultiTexCoordP3uiv = (PFNGLMULTITEXCOORDP3UIVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoordP3uiv")) == NULL) || r;
  r = ((glMultiTexCoordP4ui = (PFNGLMULTITEXCOORDP4UIPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoordP4ui")) == NULL) || r;
  r = ((glMultiTexCoordP4uiv = (PFNGLMULTITEXCOORDP4UIVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoordP4uiv")) == NULL) || r;
  r = ((glNormalP3ui = (PFNGLNORMALP3UIPROC)glewGetProcAddress((const GLubyte*)"glNormalP3ui")) == NULL) || r;
  r = ((glNormalP3uiv = (PFNGLNORMALP3UIVPROC)glewGetProcAddress((const GLubyte*)"glNormalP3uiv")) == NULL) || r;
  r = ((glSecondaryColorP3ui = (PFNGLSECONDARYCOLORP3UIPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColorP3ui")) == NULL) || r;
  r = ((glSecondaryColorP3uiv = (PFNGLSECONDARYCOLORP3UIVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColorP3uiv")) == NULL) || r;
  r = ((glTexCoordP1ui = (PFNGLTEXCOORDP1UIPROC)glewGetProcAddress((const GLubyte*)"glTexCoordP1ui")) == NULL) || r;
  r = ((glTexCoordP1uiv = (PFNGLTEXCOORDP1UIVPROC)glewGetProcAddress((const GLubyte*)"glTexCoordP1uiv")) == NULL) || r;
  r = ((glTexCoordP2ui = (PFNGLTEXCOORDP2UIPROC)glewGetProcAddress((const GLubyte*)"glTexCoordP2ui")) == NULL) || r;
  r = ((glTexCoordP2uiv = (PFNGLTEXCOORDP2UIVPROC)glewGetProcAddress((const GLubyte*)"glTexCoordP2uiv")) == NULL) || r;
  r = ((glTexCoordP3ui = (PFNGLTEXCOORDP3UIPROC)glewGetProcAddress((const GLubyte*)"glTexCoordP3ui")) == NULL) || r;
  r = ((glTexCoordP3uiv = (PFNGLTEXCOORDP3UIVPROC)glewGetProcAddress((const GLubyte*)"glTexCoordP3uiv")) == NULL) || r;
  r = ((glTexCoordP4ui = (PFNGLTEXCOORDP4UIPROC)glewGetProcAddress((const GLubyte*)"glTexCoordP4ui")) == NULL) || r;
  r = ((glTexCoordP4uiv = (PFNGLTEXCOORDP4UIVPROC)glewGetProcAddress((const GLubyte*)"glTexCoordP4uiv")) == NULL) || r;
  r = ((glVertexAttribP1ui = (PFNGLVERTEXATTRIBP1UIPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribP1ui")) == NULL) || r;
  r = ((glVertexAttribP1uiv = (PFNGLVERTEXATTRIBP1UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribP1uiv")) == NULL) || r;
  r = ((glVertexAttribP2ui = (PFNGLVERTEXATTRIBP2UIPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribP2ui")) == NULL) || r;
  r = ((glVertexAttribP2uiv = (PFNGLVERTEXATTRIBP2UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribP2uiv")) == NULL) || r;
  r = ((glVertexAttribP3ui = (PFNGLVERTEXATTRIBP3UIPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribP3ui")) == NULL) || r;
  r = ((glVertexAttribP3uiv = (PFNGLVERTEXATTRIBP3UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribP3uiv")) == NULL) || r;
  r = ((glVertexAttribP4ui = (PFNGLVERTEXATTRIBP4UIPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribP4ui")) == NULL) || r;
  r = ((glVertexAttribP4uiv = (PFNGLVERTEXATTRIBP4UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribP4uiv")) == NULL) || r;
  r = ((glVertexP2ui = (PFNGLVERTEXP2UIPROC)glewGetProcAddress((const GLubyte*)"glVertexP2ui")) == NULL) || r;
  r = ((glVertexP2uiv = (PFNGLVERTEXP2UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexP2uiv")) == NULL) || r;
  r = ((glVertexP3ui = (PFNGLVERTEXP3UIPROC)glewGetProcAddress((const GLubyte*)"glVertexP3ui")) == NULL) || r;
  r = ((glVertexP3uiv = (PFNGLVERTEXP3UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexP3uiv")) == NULL) || r;
  r = ((glVertexP4ui = (PFNGLVERTEXP4UIPROC)glewGetProcAddress((const GLubyte*)"glVertexP4ui")) == NULL) || r;
  r = ((glVertexP4uiv = (PFNGLVERTEXP4UIVPROC)glewGetProcAddress((const GLubyte*)"glVertexP4uiv")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_vertex_type_2_10_10_10_rev */

#ifdef GL_ARB_viewport_array

static GLboolean _glewInit_GL_ARB_viewport_array (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDepthRangeArrayv = (PFNGLDEPTHRANGEARRAYVPROC)glewGetProcAddress((const GLubyte*)"glDepthRangeArrayv")) == NULL) || r;
  r = ((glDepthRangeIndexed = (PFNGLDEPTHRANGEINDEXEDPROC)glewGetProcAddress((const GLubyte*)"glDepthRangeIndexed")) == NULL) || r;
  r = ((glGetDoublei_v = (PFNGLGETDOUBLEI_VPROC)glewGetProcAddress((const GLubyte*)"glGetDoublei_v")) == NULL) || r;
  r = ((glGetFloati_v = (PFNGLGETFLOATI_VPROC)glewGetProcAddress((const GLubyte*)"glGetFloati_v")) == NULL) || r;
  r = ((glScissorArrayv = (PFNGLSCISSORARRAYVPROC)glewGetProcAddress((const GLubyte*)"glScissorArrayv")) == NULL) || r;
  r = ((glScissorIndexed = (PFNGLSCISSORINDEXEDPROC)glewGetProcAddress((const GLubyte*)"glScissorIndexed")) == NULL) || r;
  r = ((glScissorIndexedv = (PFNGLSCISSORINDEXEDVPROC)glewGetProcAddress((const GLubyte*)"glScissorIndexedv")) == NULL) || r;
  r = ((glViewportArrayv = (PFNGLVIEWPORTARRAYVPROC)glewGetProcAddress((const GLubyte*)"glViewportArrayv")) == NULL) || r;
  r = ((glViewportIndexedf = (PFNGLVIEWPORTINDEXEDFPROC)glewGetProcAddress((const GLubyte*)"glViewportIndexedf")) == NULL) || r;
  r = ((glViewportIndexedfv = (PFNGLVIEWPORTINDEXEDFVPROC)glewGetProcAddress((const GLubyte*)"glViewportIndexedfv")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_viewport_array */

#ifdef GL_ARB_window_pos

static GLboolean _glewInit_GL_ARB_window_pos (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glWindowPos2dARB = (PFNGLWINDOWPOS2DARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2dARB")) == NULL) || r;
  r = ((glWindowPos2dvARB = (PFNGLWINDOWPOS2DVARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2dvARB")) == NULL) || r;
  r = ((glWindowPos2fARB = (PFNGLWINDOWPOS2FARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2fARB")) == NULL) || r;
  r = ((glWindowPos2fvARB = (PFNGLWINDOWPOS2FVARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2fvARB")) == NULL) || r;
  r = ((glWindowPos2iARB = (PFNGLWINDOWPOS2IARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2iARB")) == NULL) || r;
  r = ((glWindowPos2ivARB = (PFNGLWINDOWPOS2IVARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2ivARB")) == NULL) || r;
  r = ((glWindowPos2sARB = (PFNGLWINDOWPOS2SARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2sARB")) == NULL) || r;
  r = ((glWindowPos2svARB = (PFNGLWINDOWPOS2SVARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2svARB")) == NULL) || r;
  r = ((glWindowPos3dARB = (PFNGLWINDOWPOS3DARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3dARB")) == NULL) || r;
  r = ((glWindowPos3dvARB = (PFNGLWINDOWPOS3DVARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3dvARB")) == NULL) || r;
  r = ((glWindowPos3fARB = (PFNGLWINDOWPOS3FARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3fARB")) == NULL) || r;
  r = ((glWindowPos3fvARB = (PFNGLWINDOWPOS3FVARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3fvARB")) == NULL) || r;
  r = ((glWindowPos3iARB = (PFNGLWINDOWPOS3IARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3iARB")) == NULL) || r;
  r = ((glWindowPos3ivARB = (PFNGLWINDOWPOS3IVARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3ivARB")) == NULL) || r;
  r = ((glWindowPos3sARB = (PFNGLWINDOWPOS3SARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3sARB")) == NULL) || r;
  r = ((glWindowPos3svARB = (PFNGLWINDOWPOS3SVARBPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3svARB")) == NULL) || r;

  return r;
}

#endif /* GL_ARB_window_pos */

#ifdef GL_ATI_draw_buffers

static GLboolean _glewInit_GL_ATI_draw_buffers (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawBuffersATI = (PFNGLDRAWBUFFERSATIPROC)glewGetProcAddress((const GLubyte*)"glDrawBuffersATI")) == NULL) || r;

  return r;
}

#endif /* GL_ATI_draw_buffers */

#ifdef GL_ATI_element_array

static GLboolean _glewInit_GL_ATI_element_array (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawElementArrayATI = (PFNGLDRAWELEMENTARRAYATIPROC)glewGetProcAddress((const GLubyte*)"glDrawElementArrayATI")) == NULL) || r;
  r = ((glDrawRangeElementArrayATI = (PFNGLDRAWRANGEELEMENTARRAYATIPROC)glewGetProcAddress((const GLubyte*)"glDrawRangeElementArrayATI")) == NULL) || r;
  r = ((glElementPointerATI = (PFNGLELEMENTPOINTERATIPROC)glewGetProcAddress((const GLubyte*)"glElementPointerATI")) == NULL) || r;

  return r;
}

#endif /* GL_ATI_element_array */

#ifdef GL_ATI_envmap_bumpmap

static GLboolean _glewInit_GL_ATI_envmap_bumpmap (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetTexBumpParameterfvATI = (PFNGLGETTEXBUMPPARAMETERFVATIPROC)glewGetProcAddress((const GLubyte*)"glGetTexBumpParameterfvATI")) == NULL) || r;
  r = ((glGetTexBumpParameterivATI = (PFNGLGETTEXBUMPPARAMETERIVATIPROC)glewGetProcAddress((const GLubyte*)"glGetTexBumpParameterivATI")) == NULL) || r;
  r = ((glTexBumpParameterfvATI = (PFNGLTEXBUMPPARAMETERFVATIPROC)glewGetProcAddress((const GLubyte*)"glTexBumpParameterfvATI")) == NULL) || r;
  r = ((glTexBumpParameterivATI = (PFNGLTEXBUMPPARAMETERIVATIPROC)glewGetProcAddress((const GLubyte*)"glTexBumpParameterivATI")) == NULL) || r;

  return r;
}

#endif /* GL_ATI_envmap_bumpmap */

#ifdef GL_ATI_fragment_shader

static GLboolean _glewInit_GL_ATI_fragment_shader (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glAlphaFragmentOp1ATI = (PFNGLALPHAFRAGMENTOP1ATIPROC)glewGetProcAddress((const GLubyte*)"glAlphaFragmentOp1ATI")) == NULL) || r;
  r = ((glAlphaFragmentOp2ATI = (PFNGLALPHAFRAGMENTOP2ATIPROC)glewGetProcAddress((const GLubyte*)"glAlphaFragmentOp2ATI")) == NULL) || r;
  r = ((glAlphaFragmentOp3ATI = (PFNGLALPHAFRAGMENTOP3ATIPROC)glewGetProcAddress((const GLubyte*)"glAlphaFragmentOp3ATI")) == NULL) || r;
  r = ((glBeginFragmentShaderATI = (PFNGLBEGINFRAGMENTSHADERATIPROC)glewGetProcAddress((const GLubyte*)"glBeginFragmentShaderATI")) == NULL) || r;
  r = ((glBindFragmentShaderATI = (PFNGLBINDFRAGMENTSHADERATIPROC)glewGetProcAddress((const GLubyte*)"glBindFragmentShaderATI")) == NULL) || r;
  r = ((glColorFragmentOp1ATI = (PFNGLCOLORFRAGMENTOP1ATIPROC)glewGetProcAddress((const GLubyte*)"glColorFragmentOp1ATI")) == NULL) || r;
  r = ((glColorFragmentOp2ATI = (PFNGLCOLORFRAGMENTOP2ATIPROC)glewGetProcAddress((const GLubyte*)"glColorFragmentOp2ATI")) == NULL) || r;
  r = ((glColorFragmentOp3ATI = (PFNGLCOLORFRAGMENTOP3ATIPROC)glewGetProcAddress((const GLubyte*)"glColorFragmentOp3ATI")) == NULL) || r;
  r = ((glDeleteFragmentShaderATI = (PFNGLDELETEFRAGMENTSHADERATIPROC)glewGetProcAddress((const GLubyte*)"glDeleteFragmentShaderATI")) == NULL) || r;
  r = ((glEndFragmentShaderATI = (PFNGLENDFRAGMENTSHADERATIPROC)glewGetProcAddress((const GLubyte*)"glEndFragmentShaderATI")) == NULL) || r;
  r = ((glGenFragmentShadersATI = (PFNGLGENFRAGMENTSHADERSATIPROC)glewGetProcAddress((const GLubyte*)"glGenFragmentShadersATI")) == NULL) || r;
  r = ((glPassTexCoordATI = (PFNGLPASSTEXCOORDATIPROC)glewGetProcAddress((const GLubyte*)"glPassTexCoordATI")) == NULL) || r;
  r = ((glSampleMapATI = (PFNGLSAMPLEMAPATIPROC)glewGetProcAddress((const GLubyte*)"glSampleMapATI")) == NULL) || r;
  r = ((glSetFragmentShaderConstantATI = (PFNGLSETFRAGMENTSHADERCONSTANTATIPROC)glewGetProcAddress((const GLubyte*)"glSetFragmentShaderConstantATI")) == NULL) || r;

  return r;
}

#endif /* GL_ATI_fragment_shader */

#ifdef GL_ATI_map_object_buffer

static GLboolean _glewInit_GL_ATI_map_object_buffer (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMapObjectBufferATI = (PFNGLMAPOBJECTBUFFERATIPROC)glewGetProcAddress((const GLubyte*)"glMapObjectBufferATI")) == NULL) || r;
  r = ((glUnmapObjectBufferATI = (PFNGLUNMAPOBJECTBUFFERATIPROC)glewGetProcAddress((const GLubyte*)"glUnmapObjectBufferATI")) == NULL) || r;

  return r;
}

#endif /* GL_ATI_map_object_buffer */

#ifdef GL_ATI_pn_triangles

static GLboolean _glewInit_GL_ATI_pn_triangles (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glPNTrianglesfATI = (PFNGLPNTRIANGLESFATIPROC)glewGetProcAddress((const GLubyte*)"glPNTrianglesfATI")) == NULL) || r;
  r = ((glPNTrianglesiATI = (PFNGLPNTRIANGLESIATIPROC)glewGetProcAddress((const GLubyte*)"glPNTrianglesiATI")) == NULL) || r;

  return r;
}

#endif /* GL_ATI_pn_triangles */

#ifdef GL_ATI_separate_stencil

static GLboolean _glewInit_GL_ATI_separate_stencil (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glStencilFuncSeparateATI = (PFNGLSTENCILFUNCSEPARATEATIPROC)glewGetProcAddress((const GLubyte*)"glStencilFuncSeparateATI")) == NULL) || r;
  r = ((glStencilOpSeparateATI = (PFNGLSTENCILOPSEPARATEATIPROC)glewGetProcAddress((const GLubyte*)"glStencilOpSeparateATI")) == NULL) || r;

  return r;
}

#endif /* GL_ATI_separate_stencil */

#ifdef GL_ATI_vertex_array_object

static GLboolean _glewInit_GL_ATI_vertex_array_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glArrayObjectATI = (PFNGLARRAYOBJECTATIPROC)glewGetProcAddress((const GLubyte*)"glArrayObjectATI")) == NULL) || r;
  r = ((glFreeObjectBufferATI = (PFNGLFREEOBJECTBUFFERATIPROC)glewGetProcAddress((const GLubyte*)"glFreeObjectBufferATI")) == NULL) || r;
  r = ((glGetArrayObjectfvATI = (PFNGLGETARRAYOBJECTFVATIPROC)glewGetProcAddress((const GLubyte*)"glGetArrayObjectfvATI")) == NULL) || r;
  r = ((glGetArrayObjectivATI = (PFNGLGETARRAYOBJECTIVATIPROC)glewGetProcAddress((const GLubyte*)"glGetArrayObjectivATI")) == NULL) || r;
  r = ((glGetObjectBufferfvATI = (PFNGLGETOBJECTBUFFERFVATIPROC)glewGetProcAddress((const GLubyte*)"glGetObjectBufferfvATI")) == NULL) || r;
  r = ((glGetObjectBufferivATI = (PFNGLGETOBJECTBUFFERIVATIPROC)glewGetProcAddress((const GLubyte*)"glGetObjectBufferivATI")) == NULL) || r;
  r = ((glGetVariantArrayObjectfvATI = (PFNGLGETVARIANTARRAYOBJECTFVATIPROC)glewGetProcAddress((const GLubyte*)"glGetVariantArrayObjectfvATI")) == NULL) || r;
  r = ((glGetVariantArrayObjectivATI = (PFNGLGETVARIANTARRAYOBJECTIVATIPROC)glewGetProcAddress((const GLubyte*)"glGetVariantArrayObjectivATI")) == NULL) || r;
  r = ((glIsObjectBufferATI = (PFNGLISOBJECTBUFFERATIPROC)glewGetProcAddress((const GLubyte*)"glIsObjectBufferATI")) == NULL) || r;
  r = ((glNewObjectBufferATI = (PFNGLNEWOBJECTBUFFERATIPROC)glewGetProcAddress((const GLubyte*)"glNewObjectBufferATI")) == NULL) || r;
  r = ((glUpdateObjectBufferATI = (PFNGLUPDATEOBJECTBUFFERATIPROC)glewGetProcAddress((const GLubyte*)"glUpdateObjectBufferATI")) == NULL) || r;
  r = ((glVariantArrayObjectATI = (PFNGLVARIANTARRAYOBJECTATIPROC)glewGetProcAddress((const GLubyte*)"glVariantArrayObjectATI")) == NULL) || r;

  return r;
}

#endif /* GL_ATI_vertex_array_object */

#ifdef GL_ATI_vertex_attrib_array_object

static GLboolean _glewInit_GL_ATI_vertex_attrib_array_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetVertexAttribArrayObjectfvATI = (PFNGLGETVERTEXATTRIBARRAYOBJECTFVATIPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribArrayObjectfvATI")) == NULL) || r;
  r = ((glGetVertexAttribArrayObjectivATI = (PFNGLGETVERTEXATTRIBARRAYOBJECTIVATIPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribArrayObjectivATI")) == NULL) || r;
  r = ((glVertexAttribArrayObjectATI = (PFNGLVERTEXATTRIBARRAYOBJECTATIPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribArrayObjectATI")) == NULL) || r;

  return r;
}

#endif /* GL_ATI_vertex_attrib_array_object */

#ifdef GL_ATI_vertex_streams

static GLboolean _glewInit_GL_ATI_vertex_streams (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClientActiveVertexStreamATI = (PFNGLCLIENTACTIVEVERTEXSTREAMATIPROC)glewGetProcAddress((const GLubyte*)"glClientActiveVertexStreamATI")) == NULL) || r;
  r = ((glNormalStream3bATI = (PFNGLNORMALSTREAM3BATIPROC)glewGetProcAddress((const GLubyte*)"glNormalStream3bATI")) == NULL) || r;
  r = ((glNormalStream3bvATI = (PFNGLNORMALSTREAM3BVATIPROC)glewGetProcAddress((const GLubyte*)"glNormalStream3bvATI")) == NULL) || r;
  r = ((glNormalStream3dATI = (PFNGLNORMALSTREAM3DATIPROC)glewGetProcAddress((const GLubyte*)"glNormalStream3dATI")) == NULL) || r;
  r = ((glNormalStream3dvATI = (PFNGLNORMALSTREAM3DVATIPROC)glewGetProcAddress((const GLubyte*)"glNormalStream3dvATI")) == NULL) || r;
  r = ((glNormalStream3fATI = (PFNGLNORMALSTREAM3FATIPROC)glewGetProcAddress((const GLubyte*)"glNormalStream3fATI")) == NULL) || r;
  r = ((glNormalStream3fvATI = (PFNGLNORMALSTREAM3FVATIPROC)glewGetProcAddress((const GLubyte*)"glNormalStream3fvATI")) == NULL) || r;
  r = ((glNormalStream3iATI = (PFNGLNORMALSTREAM3IATIPROC)glewGetProcAddress((const GLubyte*)"glNormalStream3iATI")) == NULL) || r;
  r = ((glNormalStream3ivATI = (PFNGLNORMALSTREAM3IVATIPROC)glewGetProcAddress((const GLubyte*)"glNormalStream3ivATI")) == NULL) || r;
  r = ((glNormalStream3sATI = (PFNGLNORMALSTREAM3SATIPROC)glewGetProcAddress((const GLubyte*)"glNormalStream3sATI")) == NULL) || r;
  r = ((glNormalStream3svATI = (PFNGLNORMALSTREAM3SVATIPROC)glewGetProcAddress((const GLubyte*)"glNormalStream3svATI")) == NULL) || r;
  r = ((glVertexBlendEnvfATI = (PFNGLVERTEXBLENDENVFATIPROC)glewGetProcAddress((const GLubyte*)"glVertexBlendEnvfATI")) == NULL) || r;
  r = ((glVertexBlendEnviATI = (PFNGLVERTEXBLENDENVIATIPROC)glewGetProcAddress((const GLubyte*)"glVertexBlendEnviATI")) == NULL) || r;
  r = ((glVertexStream1dATI = (PFNGLVERTEXSTREAM1DATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream1dATI")) == NULL) || r;
  r = ((glVertexStream1dvATI = (PFNGLVERTEXSTREAM1DVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream1dvATI")) == NULL) || r;
  r = ((glVertexStream1fATI = (PFNGLVERTEXSTREAM1FATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream1fATI")) == NULL) || r;
  r = ((glVertexStream1fvATI = (PFNGLVERTEXSTREAM1FVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream1fvATI")) == NULL) || r;
  r = ((glVertexStream1iATI = (PFNGLVERTEXSTREAM1IATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream1iATI")) == NULL) || r;
  r = ((glVertexStream1ivATI = (PFNGLVERTEXSTREAM1IVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream1ivATI")) == NULL) || r;
  r = ((glVertexStream1sATI = (PFNGLVERTEXSTREAM1SATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream1sATI")) == NULL) || r;
  r = ((glVertexStream1svATI = (PFNGLVERTEXSTREAM1SVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream1svATI")) == NULL) || r;
  r = ((glVertexStream2dATI = (PFNGLVERTEXSTREAM2DATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream2dATI")) == NULL) || r;
  r = ((glVertexStream2dvATI = (PFNGLVERTEXSTREAM2DVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream2dvATI")) == NULL) || r;
  r = ((glVertexStream2fATI = (PFNGLVERTEXSTREAM2FATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream2fATI")) == NULL) || r;
  r = ((glVertexStream2fvATI = (PFNGLVERTEXSTREAM2FVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream2fvATI")) == NULL) || r;
  r = ((glVertexStream2iATI = (PFNGLVERTEXSTREAM2IATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream2iATI")) == NULL) || r;
  r = ((glVertexStream2ivATI = (PFNGLVERTEXSTREAM2IVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream2ivATI")) == NULL) || r;
  r = ((glVertexStream2sATI = (PFNGLVERTEXSTREAM2SATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream2sATI")) == NULL) || r;
  r = ((glVertexStream2svATI = (PFNGLVERTEXSTREAM2SVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream2svATI")) == NULL) || r;
  r = ((glVertexStream3dATI = (PFNGLVERTEXSTREAM3DATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream3dATI")) == NULL) || r;
  r = ((glVertexStream3dvATI = (PFNGLVERTEXSTREAM3DVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream3dvATI")) == NULL) || r;
  r = ((glVertexStream3fATI = (PFNGLVERTEXSTREAM3FATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream3fATI")) == NULL) || r;
  r = ((glVertexStream3fvATI = (PFNGLVERTEXSTREAM3FVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream3fvATI")) == NULL) || r;
  r = ((glVertexStream3iATI = (PFNGLVERTEXSTREAM3IATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream3iATI")) == NULL) || r;
  r = ((glVertexStream3ivATI = (PFNGLVERTEXSTREAM3IVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream3ivATI")) == NULL) || r;
  r = ((glVertexStream3sATI = (PFNGLVERTEXSTREAM3SATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream3sATI")) == NULL) || r;
  r = ((glVertexStream3svATI = (PFNGLVERTEXSTREAM3SVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream3svATI")) == NULL) || r;
  r = ((glVertexStream4dATI = (PFNGLVERTEXSTREAM4DATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream4dATI")) == NULL) || r;
  r = ((glVertexStream4dvATI = (PFNGLVERTEXSTREAM4DVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream4dvATI")) == NULL) || r;
  r = ((glVertexStream4fATI = (PFNGLVERTEXSTREAM4FATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream4fATI")) == NULL) || r;
  r = ((glVertexStream4fvATI = (PFNGLVERTEXSTREAM4FVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream4fvATI")) == NULL) || r;
  r = ((glVertexStream4iATI = (PFNGLVERTEXSTREAM4IATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream4iATI")) == NULL) || r;
  r = ((glVertexStream4ivATI = (PFNGLVERTEXSTREAM4IVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream4ivATI")) == NULL) || r;
  r = ((glVertexStream4sATI = (PFNGLVERTEXSTREAM4SATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream4sATI")) == NULL) || r;
  r = ((glVertexStream4svATI = (PFNGLVERTEXSTREAM4SVATIPROC)glewGetProcAddress((const GLubyte*)"glVertexStream4svATI")) == NULL) || r;

  return r;
}

#endif /* GL_ATI_vertex_streams */

#ifdef GL_EXT_bindable_uniform

static GLboolean _glewInit_GL_EXT_bindable_uniform (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetUniformBufferSizeEXT = (PFNGLGETUNIFORMBUFFERSIZEEXTPROC)glewGetProcAddress((const GLubyte*)"glGetUniformBufferSizeEXT")) == NULL) || r;
  r = ((glGetUniformOffsetEXT = (PFNGLGETUNIFORMOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glGetUniformOffsetEXT")) == NULL) || r;
  r = ((glUniformBufferEXT = (PFNGLUNIFORMBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glUniformBufferEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_bindable_uniform */

#ifdef GL_EXT_blend_color

static GLboolean _glewInit_GL_EXT_blend_color (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendColorEXT = (PFNGLBLENDCOLOREXTPROC)glewGetProcAddress((const GLubyte*)"glBlendColorEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_blend_color */

#ifdef GL_EXT_blend_equation_separate

static GLboolean _glewInit_GL_EXT_blend_equation_separate (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendEquationSeparateEXT = (PFNGLBLENDEQUATIONSEPARATEEXTPROC)glewGetProcAddress((const GLubyte*)"glBlendEquationSeparateEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_blend_equation_separate */

#ifdef GL_EXT_blend_func_separate

static GLboolean _glewInit_GL_EXT_blend_func_separate (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendFuncSeparateEXT = (PFNGLBLENDFUNCSEPARATEEXTPROC)glewGetProcAddress((const GLubyte*)"glBlendFuncSeparateEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_blend_func_separate */

#ifdef GL_EXT_blend_minmax

static GLboolean _glewInit_GL_EXT_blend_minmax (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendEquationEXT = (PFNGLBLENDEQUATIONEXTPROC)glewGetProcAddress((const GLubyte*)"glBlendEquationEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_blend_minmax */

#ifdef GL_EXT_color_subtable

static GLboolean _glewInit_GL_EXT_color_subtable (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glColorSubTableEXT = (PFNGLCOLORSUBTABLEEXTPROC)glewGetProcAddress((const GLubyte*)"glColorSubTableEXT")) == NULL) || r;
  r = ((glCopyColorSubTableEXT = (PFNGLCOPYCOLORSUBTABLEEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyColorSubTableEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_color_subtable */

#ifdef GL_EXT_compiled_vertex_array

static GLboolean _glewInit_GL_EXT_compiled_vertex_array (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glLockArraysEXT = (PFNGLLOCKARRAYSEXTPROC)glewGetProcAddress((const GLubyte*)"glLockArraysEXT")) == NULL) || r;
  r = ((glUnlockArraysEXT = (PFNGLUNLOCKARRAYSEXTPROC)glewGetProcAddress((const GLubyte*)"glUnlockArraysEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_compiled_vertex_array */

#ifdef GL_EXT_convolution

static GLboolean _glewInit_GL_EXT_convolution (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glConvolutionFilter1DEXT = (PFNGLCONVOLUTIONFILTER1DEXTPROC)glewGetProcAddress((const GLubyte*)"glConvolutionFilter1DEXT")) == NULL) || r;
  r = ((glConvolutionFilter2DEXT = (PFNGLCONVOLUTIONFILTER2DEXTPROC)glewGetProcAddress((const GLubyte*)"glConvolutionFilter2DEXT")) == NULL) || r;
  r = ((glConvolutionParameterfEXT = (PFNGLCONVOLUTIONPARAMETERFEXTPROC)glewGetProcAddress((const GLubyte*)"glConvolutionParameterfEXT")) == NULL) || r;
  r = ((glConvolutionParameterfvEXT = (PFNGLCONVOLUTIONPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glConvolutionParameterfvEXT")) == NULL) || r;
  r = ((glConvolutionParameteriEXT = (PFNGLCONVOLUTIONPARAMETERIEXTPROC)glewGetProcAddress((const GLubyte*)"glConvolutionParameteriEXT")) == NULL) || r;
  r = ((glConvolutionParameterivEXT = (PFNGLCONVOLUTIONPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glConvolutionParameterivEXT")) == NULL) || r;
  r = ((glCopyConvolutionFilter1DEXT = (PFNGLCOPYCONVOLUTIONFILTER1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyConvolutionFilter1DEXT")) == NULL) || r;
  r = ((glCopyConvolutionFilter2DEXT = (PFNGLCOPYCONVOLUTIONFILTER2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyConvolutionFilter2DEXT")) == NULL) || r;
  r = ((glGetConvolutionFilterEXT = (PFNGLGETCONVOLUTIONFILTEREXTPROC)glewGetProcAddress((const GLubyte*)"glGetConvolutionFilterEXT")) == NULL) || r;
  r = ((glGetConvolutionParameterfvEXT = (PFNGLGETCONVOLUTIONPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetConvolutionParameterfvEXT")) == NULL) || r;
  r = ((glGetConvolutionParameterivEXT = (PFNGLGETCONVOLUTIONPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetConvolutionParameterivEXT")) == NULL) || r;
  r = ((glGetSeparableFilterEXT = (PFNGLGETSEPARABLEFILTEREXTPROC)glewGetProcAddress((const GLubyte*)"glGetSeparableFilterEXT")) == NULL) || r;
  r = ((glSeparableFilter2DEXT = (PFNGLSEPARABLEFILTER2DEXTPROC)glewGetProcAddress((const GLubyte*)"glSeparableFilter2DEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_convolution */

#ifdef GL_EXT_coordinate_frame

static GLboolean _glewInit_GL_EXT_coordinate_frame (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBinormalPointerEXT = (PFNGLBINORMALPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glBinormalPointerEXT")) == NULL) || r;
  r = ((glTangentPointerEXT = (PFNGLTANGENTPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glTangentPointerEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_coordinate_frame */

#ifdef GL_EXT_copy_texture

static GLboolean _glewInit_GL_EXT_copy_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCopyTexImage1DEXT = (PFNGLCOPYTEXIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyTexImage1DEXT")) == NULL) || r;
  r = ((glCopyTexImage2DEXT = (PFNGLCOPYTEXIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyTexImage2DEXT")) == NULL) || r;
  r = ((glCopyTexSubImage1DEXT = (PFNGLCOPYTEXSUBIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyTexSubImage1DEXT")) == NULL) || r;
  r = ((glCopyTexSubImage2DEXT = (PFNGLCOPYTEXSUBIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyTexSubImage2DEXT")) == NULL) || r;
  r = ((glCopyTexSubImage3DEXT = (PFNGLCOPYTEXSUBIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyTexSubImage3DEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_copy_texture */

#ifdef GL_EXT_cull_vertex

static GLboolean _glewInit_GL_EXT_cull_vertex (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCullParameterdvEXT = (PFNGLCULLPARAMETERDVEXTPROC)glewGetProcAddress((const GLubyte*)"glCullParameterdvEXT")) == NULL) || r;
  r = ((glCullParameterfvEXT = (PFNGLCULLPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glCullParameterfvEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_cull_vertex */

#ifdef GL_EXT_debug_label

static GLboolean _glewInit_GL_EXT_debug_label (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetObjectLabelEXT = (PFNGLGETOBJECTLABELEXTPROC)glewGetProcAddress((const GLubyte*)"glGetObjectLabelEXT")) == NULL) || r;
  r = ((glLabelObjectEXT = (PFNGLLABELOBJECTEXTPROC)glewGetProcAddress((const GLubyte*)"glLabelObjectEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_debug_label */

#ifdef GL_EXT_debug_marker

static GLboolean _glewInit_GL_EXT_debug_marker (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glInsertEventMarkerEXT = (PFNGLINSERTEVENTMARKEREXTPROC)glewGetProcAddress((const GLubyte*)"glInsertEventMarkerEXT")) == NULL) || r;
  r = ((glPopGroupMarkerEXT = (PFNGLPOPGROUPMARKEREXTPROC)glewGetProcAddress((const GLubyte*)"glPopGroupMarkerEXT")) == NULL) || r;
  r = ((glPushGroupMarkerEXT = (PFNGLPUSHGROUPMARKEREXTPROC)glewGetProcAddress((const GLubyte*)"glPushGroupMarkerEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_debug_marker */

#ifdef GL_EXT_depth_bounds_test

static GLboolean _glewInit_GL_EXT_depth_bounds_test (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDepthBoundsEXT = (PFNGLDEPTHBOUNDSEXTPROC)glewGetProcAddress((const GLubyte*)"glDepthBoundsEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_depth_bounds_test */

#ifdef GL_EXT_direct_state_access

static GLboolean _glewInit_GL_EXT_direct_state_access (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindMultiTextureEXT = (PFNGLBINDMULTITEXTUREEXTPROC)glewGetProcAddress((const GLubyte*)"glBindMultiTextureEXT")) == NULL) || r;
  r = ((glCheckNamedFramebufferStatusEXT = (PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC)glewGetProcAddress((const GLubyte*)"glCheckNamedFramebufferStatusEXT")) == NULL) || r;
  r = ((glClientAttribDefaultEXT = (PFNGLCLIENTATTRIBDEFAULTEXTPROC)glewGetProcAddress((const GLubyte*)"glClientAttribDefaultEXT")) == NULL) || r;
  r = ((glCompressedMultiTexImage1DEXT = (PFNGLCOMPRESSEDMULTITEXIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedMultiTexImage1DEXT")) == NULL) || r;
  r = ((glCompressedMultiTexImage2DEXT = (PFNGLCOMPRESSEDMULTITEXIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedMultiTexImage2DEXT")) == NULL) || r;
  r = ((glCompressedMultiTexImage3DEXT = (PFNGLCOMPRESSEDMULTITEXIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedMultiTexImage3DEXT")) == NULL) || r;
  r = ((glCompressedMultiTexSubImage1DEXT = (PFNGLCOMPRESSEDMULTITEXSUBIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedMultiTexSubImage1DEXT")) == NULL) || r;
  r = ((glCompressedMultiTexSubImage2DEXT = (PFNGLCOMPRESSEDMULTITEXSUBIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedMultiTexSubImage2DEXT")) == NULL) || r;
  r = ((glCompressedMultiTexSubImage3DEXT = (PFNGLCOMPRESSEDMULTITEXSUBIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedMultiTexSubImage3DEXT")) == NULL) || r;
  r = ((glCompressedTextureImage1DEXT = (PFNGLCOMPRESSEDTEXTUREIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedTextureImage1DEXT")) == NULL) || r;
  r = ((glCompressedTextureImage2DEXT = (PFNGLCOMPRESSEDTEXTUREIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedTextureImage2DEXT")) == NULL) || r;
  r = ((glCompressedTextureImage3DEXT = (PFNGLCOMPRESSEDTEXTUREIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedTextureImage3DEXT")) == NULL) || r;
  r = ((glCompressedTextureSubImage1DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedTextureSubImage1DEXT")) == NULL) || r;
  r = ((glCompressedTextureSubImage2DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedTextureSubImage2DEXT")) == NULL) || r;
  r = ((glCompressedTextureSubImage3DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glCompressedTextureSubImage3DEXT")) == NULL) || r;
  r = ((glCopyMultiTexImage1DEXT = (PFNGLCOPYMULTITEXIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyMultiTexImage1DEXT")) == NULL) || r;
  r = ((glCopyMultiTexImage2DEXT = (PFNGLCOPYMULTITEXIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyMultiTexImage2DEXT")) == NULL) || r;
  r = ((glCopyMultiTexSubImage1DEXT = (PFNGLCOPYMULTITEXSUBIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyMultiTexSubImage1DEXT")) == NULL) || r;
  r = ((glCopyMultiTexSubImage2DEXT = (PFNGLCOPYMULTITEXSUBIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyMultiTexSubImage2DEXT")) == NULL) || r;
  r = ((glCopyMultiTexSubImage3DEXT = (PFNGLCOPYMULTITEXSUBIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyMultiTexSubImage3DEXT")) == NULL) || r;
  r = ((glCopyTextureImage1DEXT = (PFNGLCOPYTEXTUREIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyTextureImage1DEXT")) == NULL) || r;
  r = ((glCopyTextureImage2DEXT = (PFNGLCOPYTEXTUREIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyTextureImage2DEXT")) == NULL) || r;
  r = ((glCopyTextureSubImage1DEXT = (PFNGLCOPYTEXTURESUBIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyTextureSubImage1DEXT")) == NULL) || r;
  r = ((glCopyTextureSubImage2DEXT = (PFNGLCOPYTEXTURESUBIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyTextureSubImage2DEXT")) == NULL) || r;
  r = ((glCopyTextureSubImage3DEXT = (PFNGLCOPYTEXTURESUBIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glCopyTextureSubImage3DEXT")) == NULL) || r;
  r = ((glDisableClientStateIndexedEXT = (PFNGLDISABLECLIENTSTATEINDEXEDEXTPROC)glewGetProcAddress((const GLubyte*)"glDisableClientStateIndexedEXT")) == NULL) || r;
  r = ((glDisableClientStateiEXT = (PFNGLDISABLECLIENTSTATEIEXTPROC)glewGetProcAddress((const GLubyte*)"glDisableClientStateiEXT")) == NULL) || r;
  r = ((glDisableVertexArrayAttribEXT = (PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC)glewGetProcAddress((const GLubyte*)"glDisableVertexArrayAttribEXT")) == NULL) || r;
  r = ((glDisableVertexArrayEXT = (PFNGLDISABLEVERTEXARRAYEXTPROC)glewGetProcAddress((const GLubyte*)"glDisableVertexArrayEXT")) == NULL) || r;
  r = ((glEnableClientStateIndexedEXT = (PFNGLENABLECLIENTSTATEINDEXEDEXTPROC)glewGetProcAddress((const GLubyte*)"glEnableClientStateIndexedEXT")) == NULL) || r;
  r = ((glEnableClientStateiEXT = (PFNGLENABLECLIENTSTATEIEXTPROC)glewGetProcAddress((const GLubyte*)"glEnableClientStateiEXT")) == NULL) || r;
  r = ((glEnableVertexArrayAttribEXT = (PFNGLENABLEVERTEXARRAYATTRIBEXTPROC)glewGetProcAddress((const GLubyte*)"glEnableVertexArrayAttribEXT")) == NULL) || r;
  r = ((glEnableVertexArrayEXT = (PFNGLENABLEVERTEXARRAYEXTPROC)glewGetProcAddress((const GLubyte*)"glEnableVertexArrayEXT")) == NULL) || r;
  r = ((glFlushMappedNamedBufferRangeEXT = (PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC)glewGetProcAddress((const GLubyte*)"glFlushMappedNamedBufferRangeEXT")) == NULL) || r;
  r = ((glFramebufferDrawBufferEXT = (PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glFramebufferDrawBufferEXT")) == NULL) || r;
  r = ((glFramebufferDrawBuffersEXT = (PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC)glewGetProcAddress((const GLubyte*)"glFramebufferDrawBuffersEXT")) == NULL) || r;
  r = ((glFramebufferReadBufferEXT = (PFNGLFRAMEBUFFERREADBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glFramebufferReadBufferEXT")) == NULL) || r;
  r = ((glGenerateMultiTexMipmapEXT = (PFNGLGENERATEMULTITEXMIPMAPEXTPROC)glewGetProcAddress((const GLubyte*)"glGenerateMultiTexMipmapEXT")) == NULL) || r;
  r = ((glGenerateTextureMipmapEXT = (PFNGLGENERATETEXTUREMIPMAPEXTPROC)glewGetProcAddress((const GLubyte*)"glGenerateTextureMipmapEXT")) == NULL) || r;
  r = ((glGetCompressedMultiTexImageEXT = (PFNGLGETCOMPRESSEDMULTITEXIMAGEEXTPROC)glewGetProcAddress((const GLubyte*)"glGetCompressedMultiTexImageEXT")) == NULL) || r;
  r = ((glGetCompressedTextureImageEXT = (PFNGLGETCOMPRESSEDTEXTUREIMAGEEXTPROC)glewGetProcAddress((const GLubyte*)"glGetCompressedTextureImageEXT")) == NULL) || r;
  r = ((glGetDoubleIndexedvEXT = (PFNGLGETDOUBLEINDEXEDVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetDoubleIndexedvEXT")) == NULL) || r;
  r = ((glGetDoublei_vEXT = (PFNGLGETDOUBLEI_VEXTPROC)glewGetProcAddress((const GLubyte*)"glGetDoublei_vEXT")) == NULL) || r;
  r = ((glGetFloatIndexedvEXT = (PFNGLGETFLOATINDEXEDVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetFloatIndexedvEXT")) == NULL) || r;
  r = ((glGetFloati_vEXT = (PFNGLGETFLOATI_VEXTPROC)glewGetProcAddress((const GLubyte*)"glGetFloati_vEXT")) == NULL) || r;
  r = ((glGetFramebufferParameterivEXT = (PFNGLGETFRAMEBUFFERPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetFramebufferParameterivEXT")) == NULL) || r;
  r = ((glGetMultiTexEnvfvEXT = (PFNGLGETMULTITEXENVFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexEnvfvEXT")) == NULL) || r;
  r = ((glGetMultiTexEnvivEXT = (PFNGLGETMULTITEXENVIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexEnvivEXT")) == NULL) || r;
  r = ((glGetMultiTexGendvEXT = (PFNGLGETMULTITEXGENDVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexGendvEXT")) == NULL) || r;
  r = ((glGetMultiTexGenfvEXT = (PFNGLGETMULTITEXGENFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexGenfvEXT")) == NULL) || r;
  r = ((glGetMultiTexGenivEXT = (PFNGLGETMULTITEXGENIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexGenivEXT")) == NULL) || r;
  r = ((glGetMultiTexImageEXT = (PFNGLGETMULTITEXIMAGEEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexImageEXT")) == NULL) || r;
  r = ((glGetMultiTexLevelParameterfvEXT = (PFNGLGETMULTITEXLEVELPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexLevelParameterfvEXT")) == NULL) || r;
  r = ((glGetMultiTexLevelParameterivEXT = (PFNGLGETMULTITEXLEVELPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexLevelParameterivEXT")) == NULL) || r;
  r = ((glGetMultiTexParameterIivEXT = (PFNGLGETMULTITEXPARAMETERIIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexParameterIivEXT")) == NULL) || r;
  r = ((glGetMultiTexParameterIuivEXT = (PFNGLGETMULTITEXPARAMETERIUIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexParameterIuivEXT")) == NULL) || r;
  r = ((glGetMultiTexParameterfvEXT = (PFNGLGETMULTITEXPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexParameterfvEXT")) == NULL) || r;
  r = ((glGetMultiTexParameterivEXT = (PFNGLGETMULTITEXPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMultiTexParameterivEXT")) == NULL) || r;
  r = ((glGetNamedBufferParameterivEXT = (PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedBufferParameterivEXT")) == NULL) || r;
  r = ((glGetNamedBufferPointervEXT = (PFNGLGETNAMEDBUFFERPOINTERVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedBufferPointervEXT")) == NULL) || r;
  r = ((glGetNamedBufferSubDataEXT = (PFNGLGETNAMEDBUFFERSUBDATAEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedBufferSubDataEXT")) == NULL) || r;
  r = ((glGetNamedFramebufferAttachmentParameterivEXT = (PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedFramebufferAttachmentParameterivEXT")) == NULL) || r;
  r = ((glGetNamedProgramLocalParameterIivEXT = (PFNGLGETNAMEDPROGRAMLOCALPARAMETERIIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedProgramLocalParameterIivEXT")) == NULL) || r;
  r = ((glGetNamedProgramLocalParameterIuivEXT = (PFNGLGETNAMEDPROGRAMLOCALPARAMETERIUIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedProgramLocalParameterIuivEXT")) == NULL) || r;
  r = ((glGetNamedProgramLocalParameterdvEXT = (PFNGLGETNAMEDPROGRAMLOCALPARAMETERDVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedProgramLocalParameterdvEXT")) == NULL) || r;
  r = ((glGetNamedProgramLocalParameterfvEXT = (PFNGLGETNAMEDPROGRAMLOCALPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedProgramLocalParameterfvEXT")) == NULL) || r;
  r = ((glGetNamedProgramStringEXT = (PFNGLGETNAMEDPROGRAMSTRINGEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedProgramStringEXT")) == NULL) || r;
  r = ((glGetNamedProgramivEXT = (PFNGLGETNAMEDPROGRAMIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedProgramivEXT")) == NULL) || r;
  r = ((glGetNamedRenderbufferParameterivEXT = (PFNGLGETNAMEDRENDERBUFFERPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetNamedRenderbufferParameterivEXT")) == NULL) || r;
  r = ((glGetPointerIndexedvEXT = (PFNGLGETPOINTERINDEXEDVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetPointerIndexedvEXT")) == NULL) || r;
  r = ((glGetPointeri_vEXT = (PFNGLGETPOINTERI_VEXTPROC)glewGetProcAddress((const GLubyte*)"glGetPointeri_vEXT")) == NULL) || r;
  r = ((glGetTextureImageEXT = (PFNGLGETTEXTUREIMAGEEXTPROC)glewGetProcAddress((const GLubyte*)"glGetTextureImageEXT")) == NULL) || r;
  r = ((glGetTextureLevelParameterfvEXT = (PFNGLGETTEXTURELEVELPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetTextureLevelParameterfvEXT")) == NULL) || r;
  r = ((glGetTextureLevelParameterivEXT = (PFNGLGETTEXTURELEVELPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetTextureLevelParameterivEXT")) == NULL) || r;
  r = ((glGetTextureParameterIivEXT = (PFNGLGETTEXTUREPARAMETERIIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetTextureParameterIivEXT")) == NULL) || r;
  r = ((glGetTextureParameterIuivEXT = (PFNGLGETTEXTUREPARAMETERIUIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetTextureParameterIuivEXT")) == NULL) || r;
  r = ((glGetTextureParameterfvEXT = (PFNGLGETTEXTUREPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetTextureParameterfvEXT")) == NULL) || r;
  r = ((glGetTextureParameterivEXT = (PFNGLGETTEXTUREPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetTextureParameterivEXT")) == NULL) || r;
  r = ((glGetVertexArrayIntegeri_vEXT = (PFNGLGETVERTEXARRAYINTEGERI_VEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVertexArrayIntegeri_vEXT")) == NULL) || r;
  r = ((glGetVertexArrayIntegervEXT = (PFNGLGETVERTEXARRAYINTEGERVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVertexArrayIntegervEXT")) == NULL) || r;
  r = ((glGetVertexArrayPointeri_vEXT = (PFNGLGETVERTEXARRAYPOINTERI_VEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVertexArrayPointeri_vEXT")) == NULL) || r;
  r = ((glGetVertexArrayPointervEXT = (PFNGLGETVERTEXARRAYPOINTERVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVertexArrayPointervEXT")) == NULL) || r;
  r = ((glMapNamedBufferEXT = (PFNGLMAPNAMEDBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glMapNamedBufferEXT")) == NULL) || r;
  r = ((glMapNamedBufferRangeEXT = (PFNGLMAPNAMEDBUFFERRANGEEXTPROC)glewGetProcAddress((const GLubyte*)"glMapNamedBufferRangeEXT")) == NULL) || r;
  r = ((glMatrixFrustumEXT = (PFNGLMATRIXFRUSTUMEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixFrustumEXT")) == NULL) || r;
  r = ((glMatrixLoadIdentityEXT = (PFNGLMATRIXLOADIDENTITYEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixLoadIdentityEXT")) == NULL) || r;
  r = ((glMatrixLoadTransposedEXT = (PFNGLMATRIXLOADTRANSPOSEDEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixLoadTransposedEXT")) == NULL) || r;
  r = ((glMatrixLoadTransposefEXT = (PFNGLMATRIXLOADTRANSPOSEFEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixLoadTransposefEXT")) == NULL) || r;
  r = ((glMatrixLoaddEXT = (PFNGLMATRIXLOADDEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixLoaddEXT")) == NULL) || r;
  r = ((glMatrixLoadfEXT = (PFNGLMATRIXLOADFEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixLoadfEXT")) == NULL) || r;
  r = ((glMatrixMultTransposedEXT = (PFNGLMATRIXMULTTRANSPOSEDEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixMultTransposedEXT")) == NULL) || r;
  r = ((glMatrixMultTransposefEXT = (PFNGLMATRIXMULTTRANSPOSEFEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixMultTransposefEXT")) == NULL) || r;
  r = ((glMatrixMultdEXT = (PFNGLMATRIXMULTDEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixMultdEXT")) == NULL) || r;
  r = ((glMatrixMultfEXT = (PFNGLMATRIXMULTFEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixMultfEXT")) == NULL) || r;
  r = ((glMatrixOrthoEXT = (PFNGLMATRIXORTHOEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixOrthoEXT")) == NULL) || r;
  r = ((glMatrixPopEXT = (PFNGLMATRIXPOPEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixPopEXT")) == NULL) || r;
  r = ((glMatrixPushEXT = (PFNGLMATRIXPUSHEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixPushEXT")) == NULL) || r;
  r = ((glMatrixRotatedEXT = (PFNGLMATRIXROTATEDEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixRotatedEXT")) == NULL) || r;
  r = ((glMatrixRotatefEXT = (PFNGLMATRIXROTATEFEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixRotatefEXT")) == NULL) || r;
  r = ((glMatrixScaledEXT = (PFNGLMATRIXSCALEDEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixScaledEXT")) == NULL) || r;
  r = ((glMatrixScalefEXT = (PFNGLMATRIXSCALEFEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixScalefEXT")) == NULL) || r;
  r = ((glMatrixTranslatedEXT = (PFNGLMATRIXTRANSLATEDEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixTranslatedEXT")) == NULL) || r;
  r = ((glMatrixTranslatefEXT = (PFNGLMATRIXTRANSLATEFEXTPROC)glewGetProcAddress((const GLubyte*)"glMatrixTranslatefEXT")) == NULL) || r;
  r = ((glMultiTexBufferEXT = (PFNGLMULTITEXBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexBufferEXT")) == NULL) || r;
  r = ((glMultiTexCoordPointerEXT = (PFNGLMULTITEXCOORDPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoordPointerEXT")) == NULL) || r;
  r = ((glMultiTexEnvfEXT = (PFNGLMULTITEXENVFEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexEnvfEXT")) == NULL) || r;
  r = ((glMultiTexEnvfvEXT = (PFNGLMULTITEXENVFVEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexEnvfvEXT")) == NULL) || r;
  r = ((glMultiTexEnviEXT = (PFNGLMULTITEXENVIEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexEnviEXT")) == NULL) || r;
  r = ((glMultiTexEnvivEXT = (PFNGLMULTITEXENVIVEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexEnvivEXT")) == NULL) || r;
  r = ((glMultiTexGendEXT = (PFNGLMULTITEXGENDEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexGendEXT")) == NULL) || r;
  r = ((glMultiTexGendvEXT = (PFNGLMULTITEXGENDVEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexGendvEXT")) == NULL) || r;
  r = ((glMultiTexGenfEXT = (PFNGLMULTITEXGENFEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexGenfEXT")) == NULL) || r;
  r = ((glMultiTexGenfvEXT = (PFNGLMULTITEXGENFVEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexGenfvEXT")) == NULL) || r;
  r = ((glMultiTexGeniEXT = (PFNGLMULTITEXGENIEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexGeniEXT")) == NULL) || r;
  r = ((glMultiTexGenivEXT = (PFNGLMULTITEXGENIVEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexGenivEXT")) == NULL) || r;
  r = ((glMultiTexImage1DEXT = (PFNGLMULTITEXIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexImage1DEXT")) == NULL) || r;
  r = ((glMultiTexImage2DEXT = (PFNGLMULTITEXIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexImage2DEXT")) == NULL) || r;
  r = ((glMultiTexImage3DEXT = (PFNGLMULTITEXIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexImage3DEXT")) == NULL) || r;
  r = ((glMultiTexParameterIivEXT = (PFNGLMULTITEXPARAMETERIIVEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexParameterIivEXT")) == NULL) || r;
  r = ((glMultiTexParameterIuivEXT = (PFNGLMULTITEXPARAMETERIUIVEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexParameterIuivEXT")) == NULL) || r;
  r = ((glMultiTexParameterfEXT = (PFNGLMULTITEXPARAMETERFEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexParameterfEXT")) == NULL) || r;
  r = ((glMultiTexParameterfvEXT = (PFNGLMULTITEXPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexParameterfvEXT")) == NULL) || r;
  r = ((glMultiTexParameteriEXT = (PFNGLMULTITEXPARAMETERIEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexParameteriEXT")) == NULL) || r;
  r = ((glMultiTexParameterivEXT = (PFNGLMULTITEXPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexParameterivEXT")) == NULL) || r;
  r = ((glMultiTexRenderbufferEXT = (PFNGLMULTITEXRENDERBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexRenderbufferEXT")) == NULL) || r;
  r = ((glMultiTexSubImage1DEXT = (PFNGLMULTITEXSUBIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexSubImage1DEXT")) == NULL) || r;
  r = ((glMultiTexSubImage2DEXT = (PFNGLMULTITEXSUBIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexSubImage2DEXT")) == NULL) || r;
  r = ((glMultiTexSubImage3DEXT = (PFNGLMULTITEXSUBIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiTexSubImage3DEXT")) == NULL) || r;
  r = ((glNamedBufferDataEXT = (PFNGLNAMEDBUFFERDATAEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedBufferDataEXT")) == NULL) || r;
  r = ((glNamedBufferSubDataEXT = (PFNGLNAMEDBUFFERSUBDATAEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedBufferSubDataEXT")) == NULL) || r;
  r = ((glNamedCopyBufferSubDataEXT = (PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedCopyBufferSubDataEXT")) == NULL) || r;
  r = ((glNamedFramebufferRenderbufferEXT = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferRenderbufferEXT")) == NULL) || r;
  r = ((glNamedFramebufferTexture1DEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferTexture1DEXT")) == NULL) || r;
  r = ((glNamedFramebufferTexture2DEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferTexture2DEXT")) == NULL) || r;
  r = ((glNamedFramebufferTexture3DEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferTexture3DEXT")) == NULL) || r;
  r = ((glNamedFramebufferTextureEXT = (PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferTextureEXT")) == NULL) || r;
  r = ((glNamedFramebufferTextureFaceEXT = (PFNGLNAMEDFRAMEBUFFERTEXTUREFACEEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferTextureFaceEXT")) == NULL) || r;
  r = ((glNamedFramebufferTextureLayerEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferTextureLayerEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParameter4dEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETER4DEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParameter4dEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParameter4dvEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETER4DVEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParameter4dvEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParameter4fEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETER4FEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParameter4fEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParameter4fvEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETER4FVEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParameter4fvEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParameterI4iEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERI4IEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParameterI4iEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParameterI4ivEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERI4IVEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParameterI4ivEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParameterI4uiEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERI4UIEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParameterI4uiEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParameterI4uivEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERI4UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParameterI4uivEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParameters4fvEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERS4FVEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParameters4fvEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParametersI4ivEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERSI4IVEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParametersI4ivEXT")) == NULL) || r;
  r = ((glNamedProgramLocalParametersI4uivEXT = (PFNGLNAMEDPROGRAMLOCALPARAMETERSI4UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramLocalParametersI4uivEXT")) == NULL) || r;
  r = ((glNamedProgramStringEXT = (PFNGLNAMEDPROGRAMSTRINGEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedProgramStringEXT")) == NULL) || r;
  r = ((glNamedRenderbufferStorageEXT = (PFNGLNAMEDRENDERBUFFERSTORAGEEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedRenderbufferStorageEXT")) == NULL) || r;
  r = ((glNamedRenderbufferStorageMultisampleCoverageEXT = (PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLECOVERAGEEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedRenderbufferStorageMultisampleCoverageEXT")) == NULL) || r;
  r = ((glNamedRenderbufferStorageMultisampleEXT = (PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC)glewGetProcAddress((const GLubyte*)"glNamedRenderbufferStorageMultisampleEXT")) == NULL) || r;
  r = ((glProgramUniform1fEXT = (PFNGLPROGRAMUNIFORM1FEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1fEXT")) == NULL) || r;
  r = ((glProgramUniform1fvEXT = (PFNGLPROGRAMUNIFORM1FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1fvEXT")) == NULL) || r;
  r = ((glProgramUniform1iEXT = (PFNGLPROGRAMUNIFORM1IEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1iEXT")) == NULL) || r;
  r = ((glProgramUniform1ivEXT = (PFNGLPROGRAMUNIFORM1IVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1ivEXT")) == NULL) || r;
  r = ((glProgramUniform1uiEXT = (PFNGLPROGRAMUNIFORM1UIEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1uiEXT")) == NULL) || r;
  r = ((glProgramUniform1uivEXT = (PFNGLPROGRAMUNIFORM1UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1uivEXT")) == NULL) || r;
  r = ((glProgramUniform2fEXT = (PFNGLPROGRAMUNIFORM2FEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2fEXT")) == NULL) || r;
  r = ((glProgramUniform2fvEXT = (PFNGLPROGRAMUNIFORM2FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2fvEXT")) == NULL) || r;
  r = ((glProgramUniform2iEXT = (PFNGLPROGRAMUNIFORM2IEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2iEXT")) == NULL) || r;
  r = ((glProgramUniform2ivEXT = (PFNGLPROGRAMUNIFORM2IVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2ivEXT")) == NULL) || r;
  r = ((glProgramUniform2uiEXT = (PFNGLPROGRAMUNIFORM2UIEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2uiEXT")) == NULL) || r;
  r = ((glProgramUniform2uivEXT = (PFNGLPROGRAMUNIFORM2UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2uivEXT")) == NULL) || r;
  r = ((glProgramUniform3fEXT = (PFNGLPROGRAMUNIFORM3FEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3fEXT")) == NULL) || r;
  r = ((glProgramUniform3fvEXT = (PFNGLPROGRAMUNIFORM3FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3fvEXT")) == NULL) || r;
  r = ((glProgramUniform3iEXT = (PFNGLPROGRAMUNIFORM3IEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3iEXT")) == NULL) || r;
  r = ((glProgramUniform3ivEXT = (PFNGLPROGRAMUNIFORM3IVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3ivEXT")) == NULL) || r;
  r = ((glProgramUniform3uiEXT = (PFNGLPROGRAMUNIFORM3UIEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3uiEXT")) == NULL) || r;
  r = ((glProgramUniform3uivEXT = (PFNGLPROGRAMUNIFORM3UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3uivEXT")) == NULL) || r;
  r = ((glProgramUniform4fEXT = (PFNGLPROGRAMUNIFORM4FEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4fEXT")) == NULL) || r;
  r = ((glProgramUniform4fvEXT = (PFNGLPROGRAMUNIFORM4FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4fvEXT")) == NULL) || r;
  r = ((glProgramUniform4iEXT = (PFNGLPROGRAMUNIFORM4IEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4iEXT")) == NULL) || r;
  r = ((glProgramUniform4ivEXT = (PFNGLPROGRAMUNIFORM4IVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4ivEXT")) == NULL) || r;
  r = ((glProgramUniform4uiEXT = (PFNGLPROGRAMUNIFORM4UIEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4uiEXT")) == NULL) || r;
  r = ((glProgramUniform4uivEXT = (PFNGLPROGRAMUNIFORM4UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4uivEXT")) == NULL) || r;
  r = ((glProgramUniformMatrix2fvEXT = (PFNGLPROGRAMUNIFORMMATRIX2FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix2fvEXT")) == NULL) || r;
  r = ((glProgramUniformMatrix2x3fvEXT = (PFNGLPROGRAMUNIFORMMATRIX2X3FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix2x3fvEXT")) == NULL) || r;
  r = ((glProgramUniformMatrix2x4fvEXT = (PFNGLPROGRAMUNIFORMMATRIX2X4FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix2x4fvEXT")) == NULL) || r;
  r = ((glProgramUniformMatrix3fvEXT = (PFNGLPROGRAMUNIFORMMATRIX3FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix3fvEXT")) == NULL) || r;
  r = ((glProgramUniformMatrix3x2fvEXT = (PFNGLPROGRAMUNIFORMMATRIX3X2FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix3x2fvEXT")) == NULL) || r;
  r = ((glProgramUniformMatrix3x4fvEXT = (PFNGLPROGRAMUNIFORMMATRIX3X4FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix3x4fvEXT")) == NULL) || r;
  r = ((glProgramUniformMatrix4fvEXT = (PFNGLPROGRAMUNIFORMMATRIX4FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix4fvEXT")) == NULL) || r;
  r = ((glProgramUniformMatrix4x2fvEXT = (PFNGLPROGRAMUNIFORMMATRIX4X2FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix4x2fvEXT")) == NULL) || r;
  r = ((glProgramUniformMatrix4x3fvEXT = (PFNGLPROGRAMUNIFORMMATRIX4X3FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformMatrix4x3fvEXT")) == NULL) || r;
  r = ((glPushClientAttribDefaultEXT = (PFNGLPUSHCLIENTATTRIBDEFAULTEXTPROC)glewGetProcAddress((const GLubyte*)"glPushClientAttribDefaultEXT")) == NULL) || r;
  r = ((glTextureBufferEXT = (PFNGLTEXTUREBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glTextureBufferEXT")) == NULL) || r;
  r = ((glTextureImage1DEXT = (PFNGLTEXTUREIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureImage1DEXT")) == NULL) || r;
  r = ((glTextureImage2DEXT = (PFNGLTEXTUREIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureImage2DEXT")) == NULL) || r;
  r = ((glTextureImage3DEXT = (PFNGLTEXTUREIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureImage3DEXT")) == NULL) || r;
  r = ((glTextureParameterIivEXT = (PFNGLTEXTUREPARAMETERIIVEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureParameterIivEXT")) == NULL) || r;
  r = ((glTextureParameterIuivEXT = (PFNGLTEXTUREPARAMETERIUIVEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureParameterIuivEXT")) == NULL) || r;
  r = ((glTextureParameterfEXT = (PFNGLTEXTUREPARAMETERFEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureParameterfEXT")) == NULL) || r;
  r = ((glTextureParameterfvEXT = (PFNGLTEXTUREPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureParameterfvEXT")) == NULL) || r;
  r = ((glTextureParameteriEXT = (PFNGLTEXTUREPARAMETERIEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureParameteriEXT")) == NULL) || r;
  r = ((glTextureParameterivEXT = (PFNGLTEXTUREPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureParameterivEXT")) == NULL) || r;
  r = ((glTextureRenderbufferEXT = (PFNGLTEXTURERENDERBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glTextureRenderbufferEXT")) == NULL) || r;
  r = ((glTextureSubImage1DEXT = (PFNGLTEXTURESUBIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureSubImage1DEXT")) == NULL) || r;
  r = ((glTextureSubImage2DEXT = (PFNGLTEXTURESUBIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureSubImage2DEXT")) == NULL) || r;
  r = ((glTextureSubImage3DEXT = (PFNGLTEXTURESUBIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureSubImage3DEXT")) == NULL) || r;
  r = ((glUnmapNamedBufferEXT = (PFNGLUNMAPNAMEDBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glUnmapNamedBufferEXT")) == NULL) || r;
  r = ((glVertexArrayColorOffsetEXT = (PFNGLVERTEXARRAYCOLOROFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayColorOffsetEXT")) == NULL) || r;
  r = ((glVertexArrayEdgeFlagOffsetEXT = (PFNGLVERTEXARRAYEDGEFLAGOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayEdgeFlagOffsetEXT")) == NULL) || r;
  r = ((glVertexArrayFogCoordOffsetEXT = (PFNGLVERTEXARRAYFOGCOORDOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayFogCoordOffsetEXT")) == NULL) || r;
  r = ((glVertexArrayIndexOffsetEXT = (PFNGLVERTEXARRAYINDEXOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayIndexOffsetEXT")) == NULL) || r;
  r = ((glVertexArrayMultiTexCoordOffsetEXT = (PFNGLVERTEXARRAYMULTITEXCOORDOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayMultiTexCoordOffsetEXT")) == NULL) || r;
  r = ((glVertexArrayNormalOffsetEXT = (PFNGLVERTEXARRAYNORMALOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayNormalOffsetEXT")) == NULL) || r;
  r = ((glVertexArraySecondaryColorOffsetEXT = (PFNGLVERTEXARRAYSECONDARYCOLOROFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArraySecondaryColorOffsetEXT")) == NULL) || r;
  r = ((glVertexArrayTexCoordOffsetEXT = (PFNGLVERTEXARRAYTEXCOORDOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayTexCoordOffsetEXT")) == NULL) || r;
  r = ((glVertexArrayVertexAttribDivisorEXT = (PFNGLVERTEXARRAYVERTEXATTRIBDIVISOREXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexAttribDivisorEXT")) == NULL) || r;
  r = ((glVertexArrayVertexAttribIOffsetEXT = (PFNGLVERTEXARRAYVERTEXATTRIBIOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexAttribIOffsetEXT")) == NULL) || r;
  r = ((glVertexArrayVertexAttribOffsetEXT = (PFNGLVERTEXARRAYVERTEXATTRIBOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexAttribOffsetEXT")) == NULL) || r;
  r = ((glVertexArrayVertexOffsetEXT = (PFNGLVERTEXARRAYVERTEXOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexOffsetEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_direct_state_access */

#ifdef GL_EXT_draw_buffers2

static GLboolean _glewInit_GL_EXT_draw_buffers2 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glColorMaskIndexedEXT = (PFNGLCOLORMASKINDEXEDEXTPROC)glewGetProcAddress((const GLubyte*)"glColorMaskIndexedEXT")) == NULL) || r;
  r = ((glDisableIndexedEXT = (PFNGLDISABLEINDEXEDEXTPROC)glewGetProcAddress((const GLubyte*)"glDisableIndexedEXT")) == NULL) || r;
  r = ((glEnableIndexedEXT = (PFNGLENABLEINDEXEDEXTPROC)glewGetProcAddress((const GLubyte*)"glEnableIndexedEXT")) == NULL) || r;
  r = ((glGetBooleanIndexedvEXT = (PFNGLGETBOOLEANINDEXEDVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetBooleanIndexedvEXT")) == NULL) || r;
  r = ((glGetIntegerIndexedvEXT = (PFNGLGETINTEGERINDEXEDVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetIntegerIndexedvEXT")) == NULL) || r;
  r = ((glIsEnabledIndexedEXT = (PFNGLISENABLEDINDEXEDEXTPROC)glewGetProcAddress((const GLubyte*)"glIsEnabledIndexedEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_draw_buffers2 */

#ifdef GL_EXT_draw_instanced

static GLboolean _glewInit_GL_EXT_draw_instanced (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawArraysInstancedEXT = (PFNGLDRAWARRAYSINSTANCEDEXTPROC)glewGetProcAddress((const GLubyte*)"glDrawArraysInstancedEXT")) == NULL) || r;
  r = ((glDrawElementsInstancedEXT = (PFNGLDRAWELEMENTSINSTANCEDEXTPROC)glewGetProcAddress((const GLubyte*)"glDrawElementsInstancedEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_draw_instanced */

#ifdef GL_EXT_draw_range_elements

static GLboolean _glewInit_GL_EXT_draw_range_elements (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawRangeElementsEXT = (PFNGLDRAWRANGEELEMENTSEXTPROC)glewGetProcAddress((const GLubyte*)"glDrawRangeElementsEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_draw_range_elements */

#ifdef GL_EXT_fog_coord

static GLboolean _glewInit_GL_EXT_fog_coord (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFogCoordPointerEXT = (PFNGLFOGCOORDPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glFogCoordPointerEXT")) == NULL) || r;
  r = ((glFogCoorddEXT = (PFNGLFOGCOORDDEXTPROC)glewGetProcAddress((const GLubyte*)"glFogCoorddEXT")) == NULL) || r;
  r = ((glFogCoorddvEXT = (PFNGLFOGCOORDDVEXTPROC)glewGetProcAddress((const GLubyte*)"glFogCoorddvEXT")) == NULL) || r;
  r = ((glFogCoordfEXT = (PFNGLFOGCOORDFEXTPROC)glewGetProcAddress((const GLubyte*)"glFogCoordfEXT")) == NULL) || r;
  r = ((glFogCoordfvEXT = (PFNGLFOGCOORDFVEXTPROC)glewGetProcAddress((const GLubyte*)"glFogCoordfvEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_fog_coord */

#ifdef GL_EXT_fragment_lighting

static GLboolean _glewInit_GL_EXT_fragment_lighting (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFragmentColorMaterialEXT = (PFNGLFRAGMENTCOLORMATERIALEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentColorMaterialEXT")) == NULL) || r;
  r = ((glFragmentLightModelfEXT = (PFNGLFRAGMENTLIGHTMODELFEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightModelfEXT")) == NULL) || r;
  r = ((glFragmentLightModelfvEXT = (PFNGLFRAGMENTLIGHTMODELFVEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightModelfvEXT")) == NULL) || r;
  r = ((glFragmentLightModeliEXT = (PFNGLFRAGMENTLIGHTMODELIEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightModeliEXT")) == NULL) || r;
  r = ((glFragmentLightModelivEXT = (PFNGLFRAGMENTLIGHTMODELIVEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightModelivEXT")) == NULL) || r;
  r = ((glFragmentLightfEXT = (PFNGLFRAGMENTLIGHTFEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightfEXT")) == NULL) || r;
  r = ((glFragmentLightfvEXT = (PFNGLFRAGMENTLIGHTFVEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightfvEXT")) == NULL) || r;
  r = ((glFragmentLightiEXT = (PFNGLFRAGMENTLIGHTIEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightiEXT")) == NULL) || r;
  r = ((glFragmentLightivEXT = (PFNGLFRAGMENTLIGHTIVEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightivEXT")) == NULL) || r;
  r = ((glFragmentMaterialfEXT = (PFNGLFRAGMENTMATERIALFEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentMaterialfEXT")) == NULL) || r;
  r = ((glFragmentMaterialfvEXT = (PFNGLFRAGMENTMATERIALFVEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentMaterialfvEXT")) == NULL) || r;
  r = ((glFragmentMaterialiEXT = (PFNGLFRAGMENTMATERIALIEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentMaterialiEXT")) == NULL) || r;
  r = ((glFragmentMaterialivEXT = (PFNGLFRAGMENTMATERIALIVEXTPROC)glewGetProcAddress((const GLubyte*)"glFragmentMaterialivEXT")) == NULL) || r;
  r = ((glGetFragmentLightfvEXT = (PFNGLGETFRAGMENTLIGHTFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetFragmentLightfvEXT")) == NULL) || r;
  r = ((glGetFragmentLightivEXT = (PFNGLGETFRAGMENTLIGHTIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetFragmentLightivEXT")) == NULL) || r;
  r = ((glGetFragmentMaterialfvEXT = (PFNGLGETFRAGMENTMATERIALFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetFragmentMaterialfvEXT")) == NULL) || r;
  r = ((glGetFragmentMaterialivEXT = (PFNGLGETFRAGMENTMATERIALIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetFragmentMaterialivEXT")) == NULL) || r;
  r = ((glLightEnviEXT = (PFNGLLIGHTENVIEXTPROC)glewGetProcAddress((const GLubyte*)"glLightEnviEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_fragment_lighting */

#ifdef GL_EXT_framebuffer_blit

static GLboolean _glewInit_GL_EXT_framebuffer_blit (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlitFramebufferEXT = (PFNGLBLITFRAMEBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glBlitFramebufferEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_framebuffer_blit */

#ifdef GL_EXT_framebuffer_multisample

static GLboolean _glewInit_GL_EXT_framebuffer_multisample (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glRenderbufferStorageMultisampleEXT = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC)glewGetProcAddress((const GLubyte*)"glRenderbufferStorageMultisampleEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_framebuffer_multisample */

#ifdef GL_EXT_framebuffer_object

static GLboolean _glewInit_GL_EXT_framebuffer_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindFramebufferEXT = (PFNGLBINDFRAMEBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glBindFramebufferEXT")) == NULL) || r;
  r = ((glBindRenderbufferEXT = (PFNGLBINDRENDERBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glBindRenderbufferEXT")) == NULL) || r;
  r = ((glCheckFramebufferStatusEXT = (PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC)glewGetProcAddress((const GLubyte*)"glCheckFramebufferStatusEXT")) == NULL) || r;
  r = ((glDeleteFramebuffersEXT = (PFNGLDELETEFRAMEBUFFERSEXTPROC)glewGetProcAddress((const GLubyte*)"glDeleteFramebuffersEXT")) == NULL) || r;
  r = ((glDeleteRenderbuffersEXT = (PFNGLDELETERENDERBUFFERSEXTPROC)glewGetProcAddress((const GLubyte*)"glDeleteRenderbuffersEXT")) == NULL) || r;
  r = ((glFramebufferRenderbufferEXT = (PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glFramebufferRenderbufferEXT")) == NULL) || r;
  r = ((glFramebufferTexture1DEXT = (PFNGLFRAMEBUFFERTEXTURE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTexture1DEXT")) == NULL) || r;
  r = ((glFramebufferTexture2DEXT = (PFNGLFRAMEBUFFERTEXTURE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTexture2DEXT")) == NULL) || r;
  r = ((glFramebufferTexture3DEXT = (PFNGLFRAMEBUFFERTEXTURE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTexture3DEXT")) == NULL) || r;
  r = ((glGenFramebuffersEXT = (PFNGLGENFRAMEBUFFERSEXTPROC)glewGetProcAddress((const GLubyte*)"glGenFramebuffersEXT")) == NULL) || r;
  r = ((glGenRenderbuffersEXT = (PFNGLGENRENDERBUFFERSEXTPROC)glewGetProcAddress((const GLubyte*)"glGenRenderbuffersEXT")) == NULL) || r;
  r = ((glGenerateMipmapEXT = (PFNGLGENERATEMIPMAPEXTPROC)glewGetProcAddress((const GLubyte*)"glGenerateMipmapEXT")) == NULL) || r;
  r = ((glGetFramebufferAttachmentParameterivEXT = (PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetFramebufferAttachmentParameterivEXT")) == NULL) || r;
  r = ((glGetRenderbufferParameterivEXT = (PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetRenderbufferParameterivEXT")) == NULL) || r;
  r = ((glIsFramebufferEXT = (PFNGLISFRAMEBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glIsFramebufferEXT")) == NULL) || r;
  r = ((glIsRenderbufferEXT = (PFNGLISRENDERBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glIsRenderbufferEXT")) == NULL) || r;
  r = ((glRenderbufferStorageEXT = (PFNGLRENDERBUFFERSTORAGEEXTPROC)glewGetProcAddress((const GLubyte*)"glRenderbufferStorageEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_framebuffer_object */

#ifdef GL_EXT_geometry_shader4

static GLboolean _glewInit_GL_EXT_geometry_shader4 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFramebufferTextureEXT = (PFNGLFRAMEBUFFERTEXTUREEXTPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTextureEXT")) == NULL) || r;
  r = ((glFramebufferTextureFaceEXT = (PFNGLFRAMEBUFFERTEXTUREFACEEXTPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTextureFaceEXT")) == NULL) || r;
  r = ((glProgramParameteriEXT = (PFNGLPROGRAMPARAMETERIEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramParameteriEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_geometry_shader4 */

#ifdef GL_EXT_gpu_program_parameters

static GLboolean _glewInit_GL_EXT_gpu_program_parameters (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glProgramEnvParameters4fvEXT = (PFNGLPROGRAMENVPARAMETERS4FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParameters4fvEXT")) == NULL) || r;
  r = ((glProgramLocalParameters4fvEXT = (PFNGLPROGRAMLOCALPARAMETERS4FVEXTPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParameters4fvEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_gpu_program_parameters */

#ifdef GL_EXT_gpu_shader4

static GLboolean _glewInit_GL_EXT_gpu_shader4 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindFragDataLocationEXT = (PFNGLBINDFRAGDATALOCATIONEXTPROC)glewGetProcAddress((const GLubyte*)"glBindFragDataLocationEXT")) == NULL) || r;
  r = ((glGetFragDataLocationEXT = (PFNGLGETFRAGDATALOCATIONEXTPROC)glewGetProcAddress((const GLubyte*)"glGetFragDataLocationEXT")) == NULL) || r;
  r = ((glGetUniformuivEXT = (PFNGLGETUNIFORMUIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetUniformuivEXT")) == NULL) || r;
  r = ((glGetVertexAttribIivEXT = (PFNGLGETVERTEXATTRIBIIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribIivEXT")) == NULL) || r;
  r = ((glGetVertexAttribIuivEXT = (PFNGLGETVERTEXATTRIBIUIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribIuivEXT")) == NULL) || r;
  r = ((glUniform1uiEXT = (PFNGLUNIFORM1UIEXTPROC)glewGetProcAddress((const GLubyte*)"glUniform1uiEXT")) == NULL) || r;
  r = ((glUniform1uivEXT = (PFNGLUNIFORM1UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glUniform1uivEXT")) == NULL) || r;
  r = ((glUniform2uiEXT = (PFNGLUNIFORM2UIEXTPROC)glewGetProcAddress((const GLubyte*)"glUniform2uiEXT")) == NULL) || r;
  r = ((glUniform2uivEXT = (PFNGLUNIFORM2UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glUniform2uivEXT")) == NULL) || r;
  r = ((glUniform3uiEXT = (PFNGLUNIFORM3UIEXTPROC)glewGetProcAddress((const GLubyte*)"glUniform3uiEXT")) == NULL) || r;
  r = ((glUniform3uivEXT = (PFNGLUNIFORM3UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glUniform3uivEXT")) == NULL) || r;
  r = ((glUniform4uiEXT = (PFNGLUNIFORM4UIEXTPROC)glewGetProcAddress((const GLubyte*)"glUniform4uiEXT")) == NULL) || r;
  r = ((glUniform4uivEXT = (PFNGLUNIFORM4UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glUniform4uivEXT")) == NULL) || r;
  r = ((glVertexAttribI1iEXT = (PFNGLVERTEXATTRIBI1IEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI1iEXT")) == NULL) || r;
  r = ((glVertexAttribI1ivEXT = (PFNGLVERTEXATTRIBI1IVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI1ivEXT")) == NULL) || r;
  r = ((glVertexAttribI1uiEXT = (PFNGLVERTEXATTRIBI1UIEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI1uiEXT")) == NULL) || r;
  r = ((glVertexAttribI1uivEXT = (PFNGLVERTEXATTRIBI1UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI1uivEXT")) == NULL) || r;
  r = ((glVertexAttribI2iEXT = (PFNGLVERTEXATTRIBI2IEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI2iEXT")) == NULL) || r;
  r = ((glVertexAttribI2ivEXT = (PFNGLVERTEXATTRIBI2IVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI2ivEXT")) == NULL) || r;
  r = ((glVertexAttribI2uiEXT = (PFNGLVERTEXATTRIBI2UIEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI2uiEXT")) == NULL) || r;
  r = ((glVertexAttribI2uivEXT = (PFNGLVERTEXATTRIBI2UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI2uivEXT")) == NULL) || r;
  r = ((glVertexAttribI3iEXT = (PFNGLVERTEXATTRIBI3IEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI3iEXT")) == NULL) || r;
  r = ((glVertexAttribI3ivEXT = (PFNGLVERTEXATTRIBI3IVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI3ivEXT")) == NULL) || r;
  r = ((glVertexAttribI3uiEXT = (PFNGLVERTEXATTRIBI3UIEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI3uiEXT")) == NULL) || r;
  r = ((glVertexAttribI3uivEXT = (PFNGLVERTEXATTRIBI3UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI3uivEXT")) == NULL) || r;
  r = ((glVertexAttribI4bvEXT = (PFNGLVERTEXATTRIBI4BVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4bvEXT")) == NULL) || r;
  r = ((glVertexAttribI4iEXT = (PFNGLVERTEXATTRIBI4IEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4iEXT")) == NULL) || r;
  r = ((glVertexAttribI4ivEXT = (PFNGLVERTEXATTRIBI4IVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4ivEXT")) == NULL) || r;
  r = ((glVertexAttribI4svEXT = (PFNGLVERTEXATTRIBI4SVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4svEXT")) == NULL) || r;
  r = ((glVertexAttribI4ubvEXT = (PFNGLVERTEXATTRIBI4UBVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4ubvEXT")) == NULL) || r;
  r = ((glVertexAttribI4uiEXT = (PFNGLVERTEXATTRIBI4UIEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4uiEXT")) == NULL) || r;
  r = ((glVertexAttribI4uivEXT = (PFNGLVERTEXATTRIBI4UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4uivEXT")) == NULL) || r;
  r = ((glVertexAttribI4usvEXT = (PFNGLVERTEXATTRIBI4USVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribI4usvEXT")) == NULL) || r;
  r = ((glVertexAttribIPointerEXT = (PFNGLVERTEXATTRIBIPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribIPointerEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_gpu_shader4 */

#ifdef GL_EXT_histogram

static GLboolean _glewInit_GL_EXT_histogram (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetHistogramEXT = (PFNGLGETHISTOGRAMEXTPROC)glewGetProcAddress((const GLubyte*)"glGetHistogramEXT")) == NULL) || r;
  r = ((glGetHistogramParameterfvEXT = (PFNGLGETHISTOGRAMPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetHistogramParameterfvEXT")) == NULL) || r;
  r = ((glGetHistogramParameterivEXT = (PFNGLGETHISTOGRAMPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetHistogramParameterivEXT")) == NULL) || r;
  r = ((glGetMinmaxEXT = (PFNGLGETMINMAXEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMinmaxEXT")) == NULL) || r;
  r = ((glGetMinmaxParameterfvEXT = (PFNGLGETMINMAXPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMinmaxParameterfvEXT")) == NULL) || r;
  r = ((glGetMinmaxParameterivEXT = (PFNGLGETMINMAXPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetMinmaxParameterivEXT")) == NULL) || r;
  r = ((glHistogramEXT = (PFNGLHISTOGRAMEXTPROC)glewGetProcAddress((const GLubyte*)"glHistogramEXT")) == NULL) || r;
  r = ((glMinmaxEXT = (PFNGLMINMAXEXTPROC)glewGetProcAddress((const GLubyte*)"glMinmaxEXT")) == NULL) || r;
  r = ((glResetHistogramEXT = (PFNGLRESETHISTOGRAMEXTPROC)glewGetProcAddress((const GLubyte*)"glResetHistogramEXT")) == NULL) || r;
  r = ((glResetMinmaxEXT = (PFNGLRESETMINMAXEXTPROC)glewGetProcAddress((const GLubyte*)"glResetMinmaxEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_histogram */

#ifdef GL_EXT_index_func

static GLboolean _glewInit_GL_EXT_index_func (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glIndexFuncEXT = (PFNGLINDEXFUNCEXTPROC)glewGetProcAddress((const GLubyte*)"glIndexFuncEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_index_func */

#ifdef GL_EXT_index_material

static GLboolean _glewInit_GL_EXT_index_material (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glIndexMaterialEXT = (PFNGLINDEXMATERIALEXTPROC)glewGetProcAddress((const GLubyte*)"glIndexMaterialEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_index_material */

#ifdef GL_EXT_light_texture

static GLboolean _glewInit_GL_EXT_light_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glApplyTextureEXT = (PFNGLAPPLYTEXTUREEXTPROC)glewGetProcAddress((const GLubyte*)"glApplyTextureEXT")) == NULL) || r;
  r = ((glTextureLightEXT = (PFNGLTEXTURELIGHTEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureLightEXT")) == NULL) || r;
  r = ((glTextureMaterialEXT = (PFNGLTEXTUREMATERIALEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureMaterialEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_light_texture */

#ifdef GL_EXT_multi_draw_arrays

static GLboolean _glewInit_GL_EXT_multi_draw_arrays (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMultiDrawArraysEXT = (PFNGLMULTIDRAWARRAYSEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawArraysEXT")) == NULL) || r;
  r = ((glMultiDrawElementsEXT = (PFNGLMULTIDRAWELEMENTSEXTPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawElementsEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_multi_draw_arrays */

#ifdef GL_EXT_multisample

static GLboolean _glewInit_GL_EXT_multisample (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glSampleMaskEXT = (PFNGLSAMPLEMASKEXTPROC)glewGetProcAddress((const GLubyte*)"glSampleMaskEXT")) == NULL) || r;
  r = ((glSamplePatternEXT = (PFNGLSAMPLEPATTERNEXTPROC)glewGetProcAddress((const GLubyte*)"glSamplePatternEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_multisample */

#ifdef GL_EXT_paletted_texture

static GLboolean _glewInit_GL_EXT_paletted_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glColorTableEXT = (PFNGLCOLORTABLEEXTPROC)glewGetProcAddress((const GLubyte*)"glColorTableEXT")) == NULL) || r;
  r = ((glGetColorTableEXT = (PFNGLGETCOLORTABLEEXTPROC)glewGetProcAddress((const GLubyte*)"glGetColorTableEXT")) == NULL) || r;
  r = ((glGetColorTableParameterfvEXT = (PFNGLGETCOLORTABLEPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetColorTableParameterfvEXT")) == NULL) || r;
  r = ((glGetColorTableParameterivEXT = (PFNGLGETCOLORTABLEPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetColorTableParameterivEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_paletted_texture */

#ifdef GL_EXT_pixel_transform

static GLboolean _glewInit_GL_EXT_pixel_transform (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetPixelTransformParameterfvEXT = (PFNGLGETPIXELTRANSFORMPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetPixelTransformParameterfvEXT")) == NULL) || r;
  r = ((glGetPixelTransformParameterivEXT = (PFNGLGETPIXELTRANSFORMPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetPixelTransformParameterivEXT")) == NULL) || r;
  r = ((glPixelTransformParameterfEXT = (PFNGLPIXELTRANSFORMPARAMETERFEXTPROC)glewGetProcAddress((const GLubyte*)"glPixelTransformParameterfEXT")) == NULL) || r;
  r = ((glPixelTransformParameterfvEXT = (PFNGLPIXELTRANSFORMPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glPixelTransformParameterfvEXT")) == NULL) || r;
  r = ((glPixelTransformParameteriEXT = (PFNGLPIXELTRANSFORMPARAMETERIEXTPROC)glewGetProcAddress((const GLubyte*)"glPixelTransformParameteriEXT")) == NULL) || r;
  r = ((glPixelTransformParameterivEXT = (PFNGLPIXELTRANSFORMPARAMETERIVEXTPROC)glewGetProcAddress((const GLubyte*)"glPixelTransformParameterivEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_pixel_transform */

#ifdef GL_EXT_point_parameters

static GLboolean _glewInit_GL_EXT_point_parameters (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glPointParameterfEXT = (PFNGLPOINTPARAMETERFEXTPROC)glewGetProcAddress((const GLubyte*)"glPointParameterfEXT")) == NULL) || r;
  r = ((glPointParameterfvEXT = (PFNGLPOINTPARAMETERFVEXTPROC)glewGetProcAddress((const GLubyte*)"glPointParameterfvEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_point_parameters */

#ifdef GL_EXT_polygon_offset

static GLboolean _glewInit_GL_EXT_polygon_offset (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glPolygonOffsetEXT = (PFNGLPOLYGONOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glPolygonOffsetEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_polygon_offset */

#ifdef GL_EXT_polygon_offset_clamp

static GLboolean _glewInit_GL_EXT_polygon_offset_clamp (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glPolygonOffsetClampEXT = (PFNGLPOLYGONOFFSETCLAMPEXTPROC)glewGetProcAddress((const GLubyte*)"glPolygonOffsetClampEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_polygon_offset_clamp */

#ifdef GL_EXT_provoking_vertex

static GLboolean _glewInit_GL_EXT_provoking_vertex (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glProvokingVertexEXT = (PFNGLPROVOKINGVERTEXEXTPROC)glewGetProcAddress((const GLubyte*)"glProvokingVertexEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_provoking_vertex */

#ifdef GL_EXT_raster_multisample

static GLboolean _glewInit_GL_EXT_raster_multisample (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCoverageModulationNV = (PFNGLCOVERAGEMODULATIONNVPROC)glewGetProcAddress((const GLubyte*)"glCoverageModulationNV")) == NULL) || r;
  r = ((glCoverageModulationTableNV = (PFNGLCOVERAGEMODULATIONTABLENVPROC)glewGetProcAddress((const GLubyte*)"glCoverageModulationTableNV")) == NULL) || r;
  r = ((glGetCoverageModulationTableNV = (PFNGLGETCOVERAGEMODULATIONTABLENVPROC)glewGetProcAddress((const GLubyte*)"glGetCoverageModulationTableNV")) == NULL) || r;
  r = ((glRasterSamplesEXT = (PFNGLRASTERSAMPLESEXTPROC)glewGetProcAddress((const GLubyte*)"glRasterSamplesEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_raster_multisample */

#ifdef GL_EXT_scene_marker

static GLboolean _glewInit_GL_EXT_scene_marker (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginSceneEXT = (PFNGLBEGINSCENEEXTPROC)glewGetProcAddress((const GLubyte*)"glBeginSceneEXT")) == NULL) || r;
  r = ((glEndSceneEXT = (PFNGLENDSCENEEXTPROC)glewGetProcAddress((const GLubyte*)"glEndSceneEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_scene_marker */

#ifdef GL_EXT_secondary_color

static GLboolean _glewInit_GL_EXT_secondary_color (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glSecondaryColor3bEXT = (PFNGLSECONDARYCOLOR3BEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3bEXT")) == NULL) || r;
  r = ((glSecondaryColor3bvEXT = (PFNGLSECONDARYCOLOR3BVEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3bvEXT")) == NULL) || r;
  r = ((glSecondaryColor3dEXT = (PFNGLSECONDARYCOLOR3DEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3dEXT")) == NULL) || r;
  r = ((glSecondaryColor3dvEXT = (PFNGLSECONDARYCOLOR3DVEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3dvEXT")) == NULL) || r;
  r = ((glSecondaryColor3fEXT = (PFNGLSECONDARYCOLOR3FEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3fEXT")) == NULL) || r;
  r = ((glSecondaryColor3fvEXT = (PFNGLSECONDARYCOLOR3FVEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3fvEXT")) == NULL) || r;
  r = ((glSecondaryColor3iEXT = (PFNGLSECONDARYCOLOR3IEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3iEXT")) == NULL) || r;
  r = ((glSecondaryColor3ivEXT = (PFNGLSECONDARYCOLOR3IVEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3ivEXT")) == NULL) || r;
  r = ((glSecondaryColor3sEXT = (PFNGLSECONDARYCOLOR3SEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3sEXT")) == NULL) || r;
  r = ((glSecondaryColor3svEXT = (PFNGLSECONDARYCOLOR3SVEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3svEXT")) == NULL) || r;
  r = ((glSecondaryColor3ubEXT = (PFNGLSECONDARYCOLOR3UBEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3ubEXT")) == NULL) || r;
  r = ((glSecondaryColor3ubvEXT = (PFNGLSECONDARYCOLOR3UBVEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3ubvEXT")) == NULL) || r;
  r = ((glSecondaryColor3uiEXT = (PFNGLSECONDARYCOLOR3UIEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3uiEXT")) == NULL) || r;
  r = ((glSecondaryColor3uivEXT = (PFNGLSECONDARYCOLOR3UIVEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3uivEXT")) == NULL) || r;
  r = ((glSecondaryColor3usEXT = (PFNGLSECONDARYCOLOR3USEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3usEXT")) == NULL) || r;
  r = ((glSecondaryColor3usvEXT = (PFNGLSECONDARYCOLOR3USVEXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3usvEXT")) == NULL) || r;
  r = ((glSecondaryColorPointerEXT = (PFNGLSECONDARYCOLORPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColorPointerEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_secondary_color */

#ifdef GL_EXT_separate_shader_objects

static GLboolean _glewInit_GL_EXT_separate_shader_objects (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glActiveProgramEXT = (PFNGLACTIVEPROGRAMEXTPROC)glewGetProcAddress((const GLubyte*)"glActiveProgramEXT")) == NULL) || r;
  r = ((glCreateShaderProgramEXT = (PFNGLCREATESHADERPROGRAMEXTPROC)glewGetProcAddress((const GLubyte*)"glCreateShaderProgramEXT")) == NULL) || r;
  r = ((glUseShaderProgramEXT = (PFNGLUSESHADERPROGRAMEXTPROC)glewGetProcAddress((const GLubyte*)"glUseShaderProgramEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_separate_shader_objects */

#ifdef GL_EXT_shader_image_load_store

static GLboolean _glewInit_GL_EXT_shader_image_load_store (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindImageTextureEXT = (PFNGLBINDIMAGETEXTUREEXTPROC)glewGetProcAddress((const GLubyte*)"glBindImageTextureEXT")) == NULL) || r;
  r = ((glMemoryBarrierEXT = (PFNGLMEMORYBARRIEREXTPROC)glewGetProcAddress((const GLubyte*)"glMemoryBarrierEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_shader_image_load_store */

#ifdef GL_EXT_stencil_two_side

static GLboolean _glewInit_GL_EXT_stencil_two_side (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glActiveStencilFaceEXT = (PFNGLACTIVESTENCILFACEEXTPROC)glewGetProcAddress((const GLubyte*)"glActiveStencilFaceEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_stencil_two_side */

#ifdef GL_EXT_subtexture

static GLboolean _glewInit_GL_EXT_subtexture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexSubImage1DEXT = (PFNGLTEXSUBIMAGE1DEXTPROC)glewGetProcAddress((const GLubyte*)"glTexSubImage1DEXT")) == NULL) || r;
  r = ((glTexSubImage2DEXT = (PFNGLTEXSUBIMAGE2DEXTPROC)glewGetProcAddress((const GLubyte*)"glTexSubImage2DEXT")) == NULL) || r;
  r = ((glTexSubImage3DEXT = (PFNGLTEXSUBIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glTexSubImage3DEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_subtexture */

#ifdef GL_EXT_texture3D

static GLboolean _glewInit_GL_EXT_texture3D (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexImage3DEXT = (PFNGLTEXIMAGE3DEXTPROC)glewGetProcAddress((const GLubyte*)"glTexImage3DEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_texture3D */

#ifdef GL_EXT_texture_array

static GLboolean _glewInit_GL_EXT_texture_array (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFramebufferTextureLayerEXT = (PFNGLFRAMEBUFFERTEXTURELAYEREXTPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTextureLayerEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_texture_array */

#ifdef GL_EXT_texture_buffer_object

static GLboolean _glewInit_GL_EXT_texture_buffer_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexBufferEXT = (PFNGLTEXBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"glTexBufferEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_texture_buffer_object */

#ifdef GL_EXT_texture_integer

static GLboolean _glewInit_GL_EXT_texture_integer (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClearColorIiEXT = (PFNGLCLEARCOLORIIEXTPROC)glewGetProcAddress((const GLubyte*)"glClearColorIiEXT")) == NULL) || r;
  r = ((glClearColorIuiEXT = (PFNGLCLEARCOLORIUIEXTPROC)glewGetProcAddress((const GLubyte*)"glClearColorIuiEXT")) == NULL) || r;
  r = ((glGetTexParameterIivEXT = (PFNGLGETTEXPARAMETERIIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetTexParameterIivEXT")) == NULL) || r;
  r = ((glGetTexParameterIuivEXT = (PFNGLGETTEXPARAMETERIUIVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetTexParameterIuivEXT")) == NULL) || r;
  r = ((glTexParameterIivEXT = (PFNGLTEXPARAMETERIIVEXTPROC)glewGetProcAddress((const GLubyte*)"glTexParameterIivEXT")) == NULL) || r;
  r = ((glTexParameterIuivEXT = (PFNGLTEXPARAMETERIUIVEXTPROC)glewGetProcAddress((const GLubyte*)"glTexParameterIuivEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_texture_integer */

#ifdef GL_EXT_texture_object

static GLboolean _glewInit_GL_EXT_texture_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glAreTexturesResidentEXT = (PFNGLARETEXTURESRESIDENTEXTPROC)glewGetProcAddress((const GLubyte*)"glAreTexturesResidentEXT")) == NULL) || r;
  r = ((glBindTextureEXT = (PFNGLBINDTEXTUREEXTPROC)glewGetProcAddress((const GLubyte*)"glBindTextureEXT")) == NULL) || r;
  r = ((glDeleteTexturesEXT = (PFNGLDELETETEXTURESEXTPROC)glewGetProcAddress((const GLubyte*)"glDeleteTexturesEXT")) == NULL) || r;
  r = ((glGenTexturesEXT = (PFNGLGENTEXTURESEXTPROC)glewGetProcAddress((const GLubyte*)"glGenTexturesEXT")) == NULL) || r;
  r = ((glIsTextureEXT = (PFNGLISTEXTUREEXTPROC)glewGetProcAddress((const GLubyte*)"glIsTextureEXT")) == NULL) || r;
  r = ((glPrioritizeTexturesEXT = (PFNGLPRIORITIZETEXTURESEXTPROC)glewGetProcAddress((const GLubyte*)"glPrioritizeTexturesEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_texture_object */

#ifdef GL_EXT_texture_perturb_normal

static GLboolean _glewInit_GL_EXT_texture_perturb_normal (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTextureNormalEXT = (PFNGLTEXTURENORMALEXTPROC)glewGetProcAddress((const GLubyte*)"glTextureNormalEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_texture_perturb_normal */

#ifdef GL_EXT_timer_query

static GLboolean _glewInit_GL_EXT_timer_query (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetQueryObjecti64vEXT = (PFNGLGETQUERYOBJECTI64VEXTPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjecti64vEXT")) == NULL) || r;
  r = ((glGetQueryObjectui64vEXT = (PFNGLGETQUERYOBJECTUI64VEXTPROC)glewGetProcAddress((const GLubyte*)"glGetQueryObjectui64vEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_timer_query */

#ifdef GL_EXT_transform_feedback

static GLboolean _glewInit_GL_EXT_transform_feedback (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginTransformFeedbackEXT = (PFNGLBEGINTRANSFORMFEEDBACKEXTPROC)glewGetProcAddress((const GLubyte*)"glBeginTransformFeedbackEXT")) == NULL) || r;
  r = ((glBindBufferBaseEXT = (PFNGLBINDBUFFERBASEEXTPROC)glewGetProcAddress((const GLubyte*)"glBindBufferBaseEXT")) == NULL) || r;
  r = ((glBindBufferOffsetEXT = (PFNGLBINDBUFFEROFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glBindBufferOffsetEXT")) == NULL) || r;
  r = ((glBindBufferRangeEXT = (PFNGLBINDBUFFERRANGEEXTPROC)glewGetProcAddress((const GLubyte*)"glBindBufferRangeEXT")) == NULL) || r;
  r = ((glEndTransformFeedbackEXT = (PFNGLENDTRANSFORMFEEDBACKEXTPROC)glewGetProcAddress((const GLubyte*)"glEndTransformFeedbackEXT")) == NULL) || r;
  r = ((glGetTransformFeedbackVaryingEXT = (PFNGLGETTRANSFORMFEEDBACKVARYINGEXTPROC)glewGetProcAddress((const GLubyte*)"glGetTransformFeedbackVaryingEXT")) == NULL) || r;
  r = ((glTransformFeedbackVaryingsEXT = (PFNGLTRANSFORMFEEDBACKVARYINGSEXTPROC)glewGetProcAddress((const GLubyte*)"glTransformFeedbackVaryingsEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_transform_feedback */

#ifdef GL_EXT_vertex_array

static GLboolean _glewInit_GL_EXT_vertex_array (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glArrayElementEXT = (PFNGLARRAYELEMENTEXTPROC)glewGetProcAddress((const GLubyte*)"glArrayElementEXT")) == NULL) || r;
  r = ((glColorPointerEXT = (PFNGLCOLORPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glColorPointerEXT")) == NULL) || r;
  r = ((glDrawArraysEXT = (PFNGLDRAWARRAYSEXTPROC)glewGetProcAddress((const GLubyte*)"glDrawArraysEXT")) == NULL) || r;
  r = ((glEdgeFlagPointerEXT = (PFNGLEDGEFLAGPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glEdgeFlagPointerEXT")) == NULL) || r;
  r = ((glIndexPointerEXT = (PFNGLINDEXPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glIndexPointerEXT")) == NULL) || r;
  r = ((glNormalPointerEXT = (PFNGLNORMALPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glNormalPointerEXT")) == NULL) || r;
  r = ((glTexCoordPointerEXT = (PFNGLTEXCOORDPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glTexCoordPointerEXT")) == NULL) || r;
  r = ((glVertexPointerEXT = (PFNGLVERTEXPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glVertexPointerEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_vertex_array */

#ifdef GL_EXT_vertex_attrib_64bit

static GLboolean _glewInit_GL_EXT_vertex_attrib_64bit (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetVertexAttribLdvEXT = (PFNGLGETVERTEXATTRIBLDVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribLdvEXT")) == NULL) || r;
  r = ((glVertexArrayVertexAttribLOffsetEXT = (PFNGLVERTEXARRAYVERTEXATTRIBLOFFSETEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayVertexAttribLOffsetEXT")) == NULL) || r;
  r = ((glVertexAttribL1dEXT = (PFNGLVERTEXATTRIBL1DEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL1dEXT")) == NULL) || r;
  r = ((glVertexAttribL1dvEXT = (PFNGLVERTEXATTRIBL1DVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL1dvEXT")) == NULL) || r;
  r = ((glVertexAttribL2dEXT = (PFNGLVERTEXATTRIBL2DEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL2dEXT")) == NULL) || r;
  r = ((glVertexAttribL2dvEXT = (PFNGLVERTEXATTRIBL2DVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL2dvEXT")) == NULL) || r;
  r = ((glVertexAttribL3dEXT = (PFNGLVERTEXATTRIBL3DEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL3dEXT")) == NULL) || r;
  r = ((glVertexAttribL3dvEXT = (PFNGLVERTEXATTRIBL3DVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL3dvEXT")) == NULL) || r;
  r = ((glVertexAttribL4dEXT = (PFNGLVERTEXATTRIBL4DEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL4dEXT")) == NULL) || r;
  r = ((glVertexAttribL4dvEXT = (PFNGLVERTEXATTRIBL4DVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL4dvEXT")) == NULL) || r;
  r = ((glVertexAttribLPointerEXT = (PFNGLVERTEXATTRIBLPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribLPointerEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_vertex_attrib_64bit */

#ifdef GL_EXT_vertex_shader

static GLboolean _glewInit_GL_EXT_vertex_shader (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginVertexShaderEXT = (PFNGLBEGINVERTEXSHADEREXTPROC)glewGetProcAddress((const GLubyte*)"glBeginVertexShaderEXT")) == NULL) || r;
  r = ((glBindLightParameterEXT = (PFNGLBINDLIGHTPARAMETEREXTPROC)glewGetProcAddress((const GLubyte*)"glBindLightParameterEXT")) == NULL) || r;
  r = ((glBindMaterialParameterEXT = (PFNGLBINDMATERIALPARAMETEREXTPROC)glewGetProcAddress((const GLubyte*)"glBindMaterialParameterEXT")) == NULL) || r;
  r = ((glBindParameterEXT = (PFNGLBINDPARAMETEREXTPROC)glewGetProcAddress((const GLubyte*)"glBindParameterEXT")) == NULL) || r;
  r = ((glBindTexGenParameterEXT = (PFNGLBINDTEXGENPARAMETEREXTPROC)glewGetProcAddress((const GLubyte*)"glBindTexGenParameterEXT")) == NULL) || r;
  r = ((glBindTextureUnitParameterEXT = (PFNGLBINDTEXTUREUNITPARAMETEREXTPROC)glewGetProcAddress((const GLubyte*)"glBindTextureUnitParameterEXT")) == NULL) || r;
  r = ((glBindVertexShaderEXT = (PFNGLBINDVERTEXSHADEREXTPROC)glewGetProcAddress((const GLubyte*)"glBindVertexShaderEXT")) == NULL) || r;
  r = ((glDeleteVertexShaderEXT = (PFNGLDELETEVERTEXSHADEREXTPROC)glewGetProcAddress((const GLubyte*)"glDeleteVertexShaderEXT")) == NULL) || r;
  r = ((glDisableVariantClientStateEXT = (PFNGLDISABLEVARIANTCLIENTSTATEEXTPROC)glewGetProcAddress((const GLubyte*)"glDisableVariantClientStateEXT")) == NULL) || r;
  r = ((glEnableVariantClientStateEXT = (PFNGLENABLEVARIANTCLIENTSTATEEXTPROC)glewGetProcAddress((const GLubyte*)"glEnableVariantClientStateEXT")) == NULL) || r;
  r = ((glEndVertexShaderEXT = (PFNGLENDVERTEXSHADEREXTPROC)glewGetProcAddress((const GLubyte*)"glEndVertexShaderEXT")) == NULL) || r;
  r = ((glExtractComponentEXT = (PFNGLEXTRACTCOMPONENTEXTPROC)glewGetProcAddress((const GLubyte*)"glExtractComponentEXT")) == NULL) || r;
  r = ((glGenSymbolsEXT = (PFNGLGENSYMBOLSEXTPROC)glewGetProcAddress((const GLubyte*)"glGenSymbolsEXT")) == NULL) || r;
  r = ((glGenVertexShadersEXT = (PFNGLGENVERTEXSHADERSEXTPROC)glewGetProcAddress((const GLubyte*)"glGenVertexShadersEXT")) == NULL) || r;
  r = ((glGetInvariantBooleanvEXT = (PFNGLGETINVARIANTBOOLEANVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetInvariantBooleanvEXT")) == NULL) || r;
  r = ((glGetInvariantFloatvEXT = (PFNGLGETINVARIANTFLOATVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetInvariantFloatvEXT")) == NULL) || r;
  r = ((glGetInvariantIntegervEXT = (PFNGLGETINVARIANTINTEGERVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetInvariantIntegervEXT")) == NULL) || r;
  r = ((glGetLocalConstantBooleanvEXT = (PFNGLGETLOCALCONSTANTBOOLEANVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetLocalConstantBooleanvEXT")) == NULL) || r;
  r = ((glGetLocalConstantFloatvEXT = (PFNGLGETLOCALCONSTANTFLOATVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetLocalConstantFloatvEXT")) == NULL) || r;
  r = ((glGetLocalConstantIntegervEXT = (PFNGLGETLOCALCONSTANTINTEGERVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetLocalConstantIntegervEXT")) == NULL) || r;
  r = ((glGetVariantBooleanvEXT = (PFNGLGETVARIANTBOOLEANVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVariantBooleanvEXT")) == NULL) || r;
  r = ((glGetVariantFloatvEXT = (PFNGLGETVARIANTFLOATVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVariantFloatvEXT")) == NULL) || r;
  r = ((glGetVariantIntegervEXT = (PFNGLGETVARIANTINTEGERVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVariantIntegervEXT")) == NULL) || r;
  r = ((glGetVariantPointervEXT = (PFNGLGETVARIANTPOINTERVEXTPROC)glewGetProcAddress((const GLubyte*)"glGetVariantPointervEXT")) == NULL) || r;
  r = ((glInsertComponentEXT = (PFNGLINSERTCOMPONENTEXTPROC)glewGetProcAddress((const GLubyte*)"glInsertComponentEXT")) == NULL) || r;
  r = ((glIsVariantEnabledEXT = (PFNGLISVARIANTENABLEDEXTPROC)glewGetProcAddress((const GLubyte*)"glIsVariantEnabledEXT")) == NULL) || r;
  r = ((glSetInvariantEXT = (PFNGLSETINVARIANTEXTPROC)glewGetProcAddress((const GLubyte*)"glSetInvariantEXT")) == NULL) || r;
  r = ((glSetLocalConstantEXT = (PFNGLSETLOCALCONSTANTEXTPROC)glewGetProcAddress((const GLubyte*)"glSetLocalConstantEXT")) == NULL) || r;
  r = ((glShaderOp1EXT = (PFNGLSHADEROP1EXTPROC)glewGetProcAddress((const GLubyte*)"glShaderOp1EXT")) == NULL) || r;
  r = ((glShaderOp2EXT = (PFNGLSHADEROP2EXTPROC)glewGetProcAddress((const GLubyte*)"glShaderOp2EXT")) == NULL) || r;
  r = ((glShaderOp3EXT = (PFNGLSHADEROP3EXTPROC)glewGetProcAddress((const GLubyte*)"glShaderOp3EXT")) == NULL) || r;
  r = ((glSwizzleEXT = (PFNGLSWIZZLEEXTPROC)glewGetProcAddress((const GLubyte*)"glSwizzleEXT")) == NULL) || r;
  r = ((glVariantPointerEXT = (PFNGLVARIANTPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glVariantPointerEXT")) == NULL) || r;
  r = ((glVariantbvEXT = (PFNGLVARIANTBVEXTPROC)glewGetProcAddress((const GLubyte*)"glVariantbvEXT")) == NULL) || r;
  r = ((glVariantdvEXT = (PFNGLVARIANTDVEXTPROC)glewGetProcAddress((const GLubyte*)"glVariantdvEXT")) == NULL) || r;
  r = ((glVariantfvEXT = (PFNGLVARIANTFVEXTPROC)glewGetProcAddress((const GLubyte*)"glVariantfvEXT")) == NULL) || r;
  r = ((glVariantivEXT = (PFNGLVARIANTIVEXTPROC)glewGetProcAddress((const GLubyte*)"glVariantivEXT")) == NULL) || r;
  r = ((glVariantsvEXT = (PFNGLVARIANTSVEXTPROC)glewGetProcAddress((const GLubyte*)"glVariantsvEXT")) == NULL) || r;
  r = ((glVariantubvEXT = (PFNGLVARIANTUBVEXTPROC)glewGetProcAddress((const GLubyte*)"glVariantubvEXT")) == NULL) || r;
  r = ((glVariantuivEXT = (PFNGLVARIANTUIVEXTPROC)glewGetProcAddress((const GLubyte*)"glVariantuivEXT")) == NULL) || r;
  r = ((glVariantusvEXT = (PFNGLVARIANTUSVEXTPROC)glewGetProcAddress((const GLubyte*)"glVariantusvEXT")) == NULL) || r;
  r = ((glWriteMaskEXT = (PFNGLWRITEMASKEXTPROC)glewGetProcAddress((const GLubyte*)"glWriteMaskEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_vertex_shader */

#ifdef GL_EXT_vertex_weighting

static GLboolean _glewInit_GL_EXT_vertex_weighting (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glVertexWeightPointerEXT = (PFNGLVERTEXWEIGHTPOINTEREXTPROC)glewGetProcAddress((const GLubyte*)"glVertexWeightPointerEXT")) == NULL) || r;
  r = ((glVertexWeightfEXT = (PFNGLVERTEXWEIGHTFEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexWeightfEXT")) == NULL) || r;
  r = ((glVertexWeightfvEXT = (PFNGLVERTEXWEIGHTFVEXTPROC)glewGetProcAddress((const GLubyte*)"glVertexWeightfvEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_vertex_weighting */

#ifdef GL_EXT_x11_sync_object

static GLboolean _glewInit_GL_EXT_x11_sync_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glImportSyncEXT = (PFNGLIMPORTSYNCEXTPROC)glewGetProcAddress((const GLubyte*)"glImportSyncEXT")) == NULL) || r;

  return r;
}

#endif /* GL_EXT_x11_sync_object */

#ifdef GL_GREMEDY_frame_terminator

static GLboolean _glewInit_GL_GREMEDY_frame_terminator (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFrameTerminatorGREMEDY = (PFNGLFRAMETERMINATORGREMEDYPROC)glewGetProcAddress((const GLubyte*)"glFrameTerminatorGREMEDY")) == NULL) || r;

  return r;
}

#endif /* GL_GREMEDY_frame_terminator */

#ifdef GL_GREMEDY_string_marker

static GLboolean _glewInit_GL_GREMEDY_string_marker (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glStringMarkerGREMEDY = (PFNGLSTRINGMARKERGREMEDYPROC)glewGetProcAddress((const GLubyte*)"glStringMarkerGREMEDY")) == NULL) || r;

  return r;
}

#endif /* GL_GREMEDY_string_marker */

#ifdef GL_HP_image_transform

static GLboolean _glewInit_GL_HP_image_transform (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetImageTransformParameterfvHP = (PFNGLGETIMAGETRANSFORMPARAMETERFVHPPROC)glewGetProcAddress((const GLubyte*)"glGetImageTransformParameterfvHP")) == NULL) || r;
  r = ((glGetImageTransformParameterivHP = (PFNGLGETIMAGETRANSFORMPARAMETERIVHPPROC)glewGetProcAddress((const GLubyte*)"glGetImageTransformParameterivHP")) == NULL) || r;
  r = ((glImageTransformParameterfHP = (PFNGLIMAGETRANSFORMPARAMETERFHPPROC)glewGetProcAddress((const GLubyte*)"glImageTransformParameterfHP")) == NULL) || r;
  r = ((glImageTransformParameterfvHP = (PFNGLIMAGETRANSFORMPARAMETERFVHPPROC)glewGetProcAddress((const GLubyte*)"glImageTransformParameterfvHP")) == NULL) || r;
  r = ((glImageTransformParameteriHP = (PFNGLIMAGETRANSFORMPARAMETERIHPPROC)glewGetProcAddress((const GLubyte*)"glImageTransformParameteriHP")) == NULL) || r;
  r = ((glImageTransformParameterivHP = (PFNGLIMAGETRANSFORMPARAMETERIVHPPROC)glewGetProcAddress((const GLubyte*)"glImageTransformParameterivHP")) == NULL) || r;

  return r;
}

#endif /* GL_HP_image_transform */

#ifdef GL_IBM_multimode_draw_arrays

static GLboolean _glewInit_GL_IBM_multimode_draw_arrays (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMultiModeDrawArraysIBM = (PFNGLMULTIMODEDRAWARRAYSIBMPROC)glewGetProcAddress((const GLubyte*)"glMultiModeDrawArraysIBM")) == NULL) || r;
  r = ((glMultiModeDrawElementsIBM = (PFNGLMULTIMODEDRAWELEMENTSIBMPROC)glewGetProcAddress((const GLubyte*)"glMultiModeDrawElementsIBM")) == NULL) || r;

  return r;
}

#endif /* GL_IBM_multimode_draw_arrays */

#ifdef GL_IBM_vertex_array_lists

static GLboolean _glewInit_GL_IBM_vertex_array_lists (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glColorPointerListIBM = (PFNGLCOLORPOINTERLISTIBMPROC)glewGetProcAddress((const GLubyte*)"glColorPointerListIBM")) == NULL) || r;
  r = ((glEdgeFlagPointerListIBM = (PFNGLEDGEFLAGPOINTERLISTIBMPROC)glewGetProcAddress((const GLubyte*)"glEdgeFlagPointerListIBM")) == NULL) || r;
  r = ((glFogCoordPointerListIBM = (PFNGLFOGCOORDPOINTERLISTIBMPROC)glewGetProcAddress((const GLubyte*)"glFogCoordPointerListIBM")) == NULL) || r;
  r = ((glIndexPointerListIBM = (PFNGLINDEXPOINTERLISTIBMPROC)glewGetProcAddress((const GLubyte*)"glIndexPointerListIBM")) == NULL) || r;
  r = ((glNormalPointerListIBM = (PFNGLNORMALPOINTERLISTIBMPROC)glewGetProcAddress((const GLubyte*)"glNormalPointerListIBM")) == NULL) || r;
  r = ((glSecondaryColorPointerListIBM = (PFNGLSECONDARYCOLORPOINTERLISTIBMPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColorPointerListIBM")) == NULL) || r;
  r = ((glTexCoordPointerListIBM = (PFNGLTEXCOORDPOINTERLISTIBMPROC)glewGetProcAddress((const GLubyte*)"glTexCoordPointerListIBM")) == NULL) || r;
  r = ((glVertexPointerListIBM = (PFNGLVERTEXPOINTERLISTIBMPROC)glewGetProcAddress((const GLubyte*)"glVertexPointerListIBM")) == NULL) || r;

  return r;
}

#endif /* GL_IBM_vertex_array_lists */

#ifdef GL_INTEL_map_texture

static GLboolean _glewInit_GL_INTEL_map_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMapTexture2DINTEL = (PFNGLMAPTEXTURE2DINTELPROC)glewGetProcAddress((const GLubyte*)"glMapTexture2DINTEL")) == NULL) || r;
  r = ((glSyncTextureINTEL = (PFNGLSYNCTEXTUREINTELPROC)glewGetProcAddress((const GLubyte*)"glSyncTextureINTEL")) == NULL) || r;
  r = ((glUnmapTexture2DINTEL = (PFNGLUNMAPTEXTURE2DINTELPROC)glewGetProcAddress((const GLubyte*)"glUnmapTexture2DINTEL")) == NULL) || r;

  return r;
}

#endif /* GL_INTEL_map_texture */

#ifdef GL_INTEL_parallel_arrays

static GLboolean _glewInit_GL_INTEL_parallel_arrays (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glColorPointervINTEL = (PFNGLCOLORPOINTERVINTELPROC)glewGetProcAddress((const GLubyte*)"glColorPointervINTEL")) == NULL) || r;
  r = ((glNormalPointervINTEL = (PFNGLNORMALPOINTERVINTELPROC)glewGetProcAddress((const GLubyte*)"glNormalPointervINTEL")) == NULL) || r;
  r = ((glTexCoordPointervINTEL = (PFNGLTEXCOORDPOINTERVINTELPROC)glewGetProcAddress((const GLubyte*)"glTexCoordPointervINTEL")) == NULL) || r;
  r = ((glVertexPointervINTEL = (PFNGLVERTEXPOINTERVINTELPROC)glewGetProcAddress((const GLubyte*)"glVertexPointervINTEL")) == NULL) || r;

  return r;
}

#endif /* GL_INTEL_parallel_arrays */

#ifdef GL_INTEL_performance_query

static GLboolean _glewInit_GL_INTEL_performance_query (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginPerfQueryINTEL = (PFNGLBEGINPERFQUERYINTELPROC)glewGetProcAddress((const GLubyte*)"glBeginPerfQueryINTEL")) == NULL) || r;
  r = ((glCreatePerfQueryINTEL = (PFNGLCREATEPERFQUERYINTELPROC)glewGetProcAddress((const GLubyte*)"glCreatePerfQueryINTEL")) == NULL) || r;
  r = ((glDeletePerfQueryINTEL = (PFNGLDELETEPERFQUERYINTELPROC)glewGetProcAddress((const GLubyte*)"glDeletePerfQueryINTEL")) == NULL) || r;
  r = ((glEndPerfQueryINTEL = (PFNGLENDPERFQUERYINTELPROC)glewGetProcAddress((const GLubyte*)"glEndPerfQueryINTEL")) == NULL) || r;
  r = ((glGetFirstPerfQueryIdINTEL = (PFNGLGETFIRSTPERFQUERYIDINTELPROC)glewGetProcAddress((const GLubyte*)"glGetFirstPerfQueryIdINTEL")) == NULL) || r;
  r = ((glGetNextPerfQueryIdINTEL = (PFNGLGETNEXTPERFQUERYIDINTELPROC)glewGetProcAddress((const GLubyte*)"glGetNextPerfQueryIdINTEL")) == NULL) || r;
  r = ((glGetPerfCounterInfoINTEL = (PFNGLGETPERFCOUNTERINFOINTELPROC)glewGetProcAddress((const GLubyte*)"glGetPerfCounterInfoINTEL")) == NULL) || r;
  r = ((glGetPerfQueryDataINTEL = (PFNGLGETPERFQUERYDATAINTELPROC)glewGetProcAddress((const GLubyte*)"glGetPerfQueryDataINTEL")) == NULL) || r;
  r = ((glGetPerfQueryIdByNameINTEL = (PFNGLGETPERFQUERYIDBYNAMEINTELPROC)glewGetProcAddress((const GLubyte*)"glGetPerfQueryIdByNameINTEL")) == NULL) || r;
  r = ((glGetPerfQueryInfoINTEL = (PFNGLGETPERFQUERYINFOINTELPROC)glewGetProcAddress((const GLubyte*)"glGetPerfQueryInfoINTEL")) == NULL) || r;

  return r;
}

#endif /* GL_INTEL_performance_query */

#ifdef GL_INTEL_texture_scissor

static GLboolean _glewInit_GL_INTEL_texture_scissor (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexScissorFuncINTEL = (PFNGLTEXSCISSORFUNCINTELPROC)glewGetProcAddress((const GLubyte*)"glTexScissorFuncINTEL")) == NULL) || r;
  r = ((glTexScissorINTEL = (PFNGLTEXSCISSORINTELPROC)glewGetProcAddress((const GLubyte*)"glTexScissorINTEL")) == NULL) || r;

  return r;
}

#endif /* GL_INTEL_texture_scissor */

#ifdef GL_KHR_blend_equation_advanced

static GLboolean _glewInit_GL_KHR_blend_equation_advanced (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendBarrierKHR = (PFNGLBLENDBARRIERKHRPROC)glewGetProcAddress((const GLubyte*)"glBlendBarrierKHR")) == NULL) || r;

  return r;
}

#endif /* GL_KHR_blend_equation_advanced */

#ifdef GL_KHR_debug

static GLboolean _glewInit_GL_KHR_debug (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDebugMessageCallback = (PFNGLDEBUGMESSAGECALLBACKPROC)glewGetProcAddress((const GLubyte*)"glDebugMessageCallback")) == NULL) || r;
  r = ((glDebugMessageControl = (PFNGLDEBUGMESSAGECONTROLPROC)glewGetProcAddress((const GLubyte*)"glDebugMessageControl")) == NULL) || r;
  r = ((glDebugMessageInsert = (PFNGLDEBUGMESSAGEINSERTPROC)glewGetProcAddress((const GLubyte*)"glDebugMessageInsert")) == NULL) || r;
  r = ((glGetDebugMessageLog = (PFNGLGETDEBUGMESSAGELOGPROC)glewGetProcAddress((const GLubyte*)"glGetDebugMessageLog")) == NULL) || r;
  r = ((glGetObjectLabel = (PFNGLGETOBJECTLABELPROC)glewGetProcAddress((const GLubyte*)"glGetObjectLabel")) == NULL) || r;
  r = ((glGetObjectPtrLabel = (PFNGLGETOBJECTPTRLABELPROC)glewGetProcAddress((const GLubyte*)"glGetObjectPtrLabel")) == NULL) || r;
  r = ((glObjectLabel = (PFNGLOBJECTLABELPROC)glewGetProcAddress((const GLubyte*)"glObjectLabel")) == NULL) || r;
  r = ((glObjectPtrLabel = (PFNGLOBJECTPTRLABELPROC)glewGetProcAddress((const GLubyte*)"glObjectPtrLabel")) == NULL) || r;
  r = ((glPopDebugGroup = (PFNGLPOPDEBUGGROUPPROC)glewGetProcAddress((const GLubyte*)"glPopDebugGroup")) == NULL) || r;
  r = ((glPushDebugGroup = (PFNGLPUSHDEBUGGROUPPROC)glewGetProcAddress((const GLubyte*)"glPushDebugGroup")) == NULL) || r;

  return r;
}

#endif /* GL_KHR_debug */

#ifdef GL_KHR_robustness

static GLboolean _glewInit_GL_KHR_robustness (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetnUniformfv = (PFNGLGETNUNIFORMFVPROC)glewGetProcAddress((const GLubyte*)"glGetnUniformfv")) == NULL) || r;
  r = ((glGetnUniformiv = (PFNGLGETNUNIFORMIVPROC)glewGetProcAddress((const GLubyte*)"glGetnUniformiv")) == NULL) || r;
  r = ((glGetnUniformuiv = (PFNGLGETNUNIFORMUIVPROC)glewGetProcAddress((const GLubyte*)"glGetnUniformuiv")) == NULL) || r;
  r = ((glReadnPixels = (PFNGLREADNPIXELSPROC)glewGetProcAddress((const GLubyte*)"glReadnPixels")) == NULL) || r;

  return r;
}

#endif /* GL_KHR_robustness */

#ifdef GL_KTX_buffer_region

static GLboolean _glewInit_GL_KTX_buffer_region (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBufferRegionEnabled = (PFNGLBUFFERREGIONENABLEDPROC)glewGetProcAddress((const GLubyte*)"glBufferRegionEnabled")) == NULL) || r;
  r = ((glDeleteBufferRegion = (PFNGLDELETEBUFFERREGIONPROC)glewGetProcAddress((const GLubyte*)"glDeleteBufferRegion")) == NULL) || r;
  r = ((glDrawBufferRegion = (PFNGLDRAWBUFFERREGIONPROC)glewGetProcAddress((const GLubyte*)"glDrawBufferRegion")) == NULL) || r;
  r = ((glNewBufferRegion = (PFNGLNEWBUFFERREGIONPROC)glewGetProcAddress((const GLubyte*)"glNewBufferRegion")) == NULL) || r;
  r = ((glReadBufferRegion = (PFNGLREADBUFFERREGIONPROC)glewGetProcAddress((const GLubyte*)"glReadBufferRegion")) == NULL) || r;

  return r;
}

#endif /* GL_KTX_buffer_region */

#ifdef GL_MESA_resize_buffers

static GLboolean _glewInit_GL_MESA_resize_buffers (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glResizeBuffersMESA = (PFNGLRESIZEBUFFERSMESAPROC)glewGetProcAddress((const GLubyte*)"glResizeBuffersMESA")) == NULL) || r;

  return r;
}

#endif /* GL_MESA_resize_buffers */

#ifdef GL_MESA_window_pos

static GLboolean _glewInit_GL_MESA_window_pos (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glWindowPos2dMESA = (PFNGLWINDOWPOS2DMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2dMESA")) == NULL) || r;
  r = ((glWindowPos2dvMESA = (PFNGLWINDOWPOS2DVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2dvMESA")) == NULL) || r;
  r = ((glWindowPos2fMESA = (PFNGLWINDOWPOS2FMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2fMESA")) == NULL) || r;
  r = ((glWindowPos2fvMESA = (PFNGLWINDOWPOS2FVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2fvMESA")) == NULL) || r;
  r = ((glWindowPos2iMESA = (PFNGLWINDOWPOS2IMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2iMESA")) == NULL) || r;
  r = ((glWindowPos2ivMESA = (PFNGLWINDOWPOS2IVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2ivMESA")) == NULL) || r;
  r = ((glWindowPos2sMESA = (PFNGLWINDOWPOS2SMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2sMESA")) == NULL) || r;
  r = ((glWindowPos2svMESA = (PFNGLWINDOWPOS2SVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos2svMESA")) == NULL) || r;
  r = ((glWindowPos3dMESA = (PFNGLWINDOWPOS3DMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3dMESA")) == NULL) || r;
  r = ((glWindowPos3dvMESA = (PFNGLWINDOWPOS3DVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3dvMESA")) == NULL) || r;
  r = ((glWindowPos3fMESA = (PFNGLWINDOWPOS3FMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3fMESA")) == NULL) || r;
  r = ((glWindowPos3fvMESA = (PFNGLWINDOWPOS3FVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3fvMESA")) == NULL) || r;
  r = ((glWindowPos3iMESA = (PFNGLWINDOWPOS3IMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3iMESA")) == NULL) || r;
  r = ((glWindowPos3ivMESA = (PFNGLWINDOWPOS3IVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3ivMESA")) == NULL) || r;
  r = ((glWindowPos3sMESA = (PFNGLWINDOWPOS3SMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3sMESA")) == NULL) || r;
  r = ((glWindowPos3svMESA = (PFNGLWINDOWPOS3SVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos3svMESA")) == NULL) || r;
  r = ((glWindowPos4dMESA = (PFNGLWINDOWPOS4DMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos4dMESA")) == NULL) || r;
  r = ((glWindowPos4dvMESA = (PFNGLWINDOWPOS4DVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos4dvMESA")) == NULL) || r;
  r = ((glWindowPos4fMESA = (PFNGLWINDOWPOS4FMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos4fMESA")) == NULL) || r;
  r = ((glWindowPos4fvMESA = (PFNGLWINDOWPOS4FVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos4fvMESA")) == NULL) || r;
  r = ((glWindowPos4iMESA = (PFNGLWINDOWPOS4IMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos4iMESA")) == NULL) || r;
  r = ((glWindowPos4ivMESA = (PFNGLWINDOWPOS4IVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos4ivMESA")) == NULL) || r;
  r = ((glWindowPos4sMESA = (PFNGLWINDOWPOS4SMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos4sMESA")) == NULL) || r;
  r = ((glWindowPos4svMESA = (PFNGLWINDOWPOS4SVMESAPROC)glewGetProcAddress((const GLubyte*)"glWindowPos4svMESA")) == NULL) || r;

  return r;
}

#endif /* GL_MESA_window_pos */

#ifdef GL_NVX_conditional_render

static GLboolean _glewInit_GL_NVX_conditional_render (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginConditionalRenderNVX = (PFNGLBEGINCONDITIONALRENDERNVXPROC)glewGetProcAddress((const GLubyte*)"glBeginConditionalRenderNVX")) == NULL) || r;
  r = ((glEndConditionalRenderNVX = (PFNGLENDCONDITIONALRENDERNVXPROC)glewGetProcAddress((const GLubyte*)"glEndConditionalRenderNVX")) == NULL) || r;

  return r;
}

#endif /* GL_NVX_conditional_render */

#ifdef GL_NV_bindless_multi_draw_indirect

static GLboolean _glewInit_GL_NV_bindless_multi_draw_indirect (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMultiDrawArraysIndirectBindlessNV = (PFNGLMULTIDRAWARRAYSINDIRECTBINDLESSNVPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawArraysIndirectBindlessNV")) == NULL) || r;
  r = ((glMultiDrawElementsIndirectBindlessNV = (PFNGLMULTIDRAWELEMENTSINDIRECTBINDLESSNVPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawElementsIndirectBindlessNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_bindless_multi_draw_indirect */

#ifdef GL_NV_bindless_multi_draw_indirect_count

static GLboolean _glewInit_GL_NV_bindless_multi_draw_indirect_count (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glMultiDrawArraysIndirectBindlessCountNV = (PFNGLMULTIDRAWARRAYSINDIRECTBINDLESSCOUNTNVPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawArraysIndirectBindlessCountNV")) == NULL) || r;
  r = ((glMultiDrawElementsIndirectBindlessCountNV = (PFNGLMULTIDRAWELEMENTSINDIRECTBINDLESSCOUNTNVPROC)glewGetProcAddress((const GLubyte*)"glMultiDrawElementsIndirectBindlessCountNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_bindless_multi_draw_indirect_count */

#ifdef GL_NV_bindless_texture

static GLboolean _glewInit_GL_NV_bindless_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetImageHandleNV = (PFNGLGETIMAGEHANDLENVPROC)glewGetProcAddress((const GLubyte*)"glGetImageHandleNV")) == NULL) || r;
  r = ((glGetTextureHandleNV = (PFNGLGETTEXTUREHANDLENVPROC)glewGetProcAddress((const GLubyte*)"glGetTextureHandleNV")) == NULL) || r;
  r = ((glGetTextureSamplerHandleNV = (PFNGLGETTEXTURESAMPLERHANDLENVPROC)glewGetProcAddress((const GLubyte*)"glGetTextureSamplerHandleNV")) == NULL) || r;
  r = ((glIsImageHandleResidentNV = (PFNGLISIMAGEHANDLERESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glIsImageHandleResidentNV")) == NULL) || r;
  r = ((glIsTextureHandleResidentNV = (PFNGLISTEXTUREHANDLERESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glIsTextureHandleResidentNV")) == NULL) || r;
  r = ((glMakeImageHandleNonResidentNV = (PFNGLMAKEIMAGEHANDLENONRESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glMakeImageHandleNonResidentNV")) == NULL) || r;
  r = ((glMakeImageHandleResidentNV = (PFNGLMAKEIMAGEHANDLERESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glMakeImageHandleResidentNV")) == NULL) || r;
  r = ((glMakeTextureHandleNonResidentNV = (PFNGLMAKETEXTUREHANDLENONRESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glMakeTextureHandleNonResidentNV")) == NULL) || r;
  r = ((glMakeTextureHandleResidentNV = (PFNGLMAKETEXTUREHANDLERESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glMakeTextureHandleResidentNV")) == NULL) || r;
  r = ((glProgramUniformHandleui64NV = (PFNGLPROGRAMUNIFORMHANDLEUI64NVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformHandleui64NV")) == NULL) || r;
  r = ((glProgramUniformHandleui64vNV = (PFNGLPROGRAMUNIFORMHANDLEUI64VNVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformHandleui64vNV")) == NULL) || r;
  r = ((glUniformHandleui64NV = (PFNGLUNIFORMHANDLEUI64NVPROC)glewGetProcAddress((const GLubyte*)"glUniformHandleui64NV")) == NULL) || r;
  r = ((glUniformHandleui64vNV = (PFNGLUNIFORMHANDLEUI64VNVPROC)glewGetProcAddress((const GLubyte*)"glUniformHandleui64vNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_bindless_texture */

#ifdef GL_NV_blend_equation_advanced

static GLboolean _glewInit_GL_NV_blend_equation_advanced (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBlendBarrierNV = (PFNGLBLENDBARRIERNVPROC)glewGetProcAddress((const GLubyte*)"glBlendBarrierNV")) == NULL) || r;
  r = ((glBlendParameteriNV = (PFNGLBLENDPARAMETERINVPROC)glewGetProcAddress((const GLubyte*)"glBlendParameteriNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_blend_equation_advanced */

#ifdef GL_NV_conditional_render

static GLboolean _glewInit_GL_NV_conditional_render (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginConditionalRenderNV = (PFNGLBEGINCONDITIONALRENDERNVPROC)glewGetProcAddress((const GLubyte*)"glBeginConditionalRenderNV")) == NULL) || r;
  r = ((glEndConditionalRenderNV = (PFNGLENDCONDITIONALRENDERNVPROC)glewGetProcAddress((const GLubyte*)"glEndConditionalRenderNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_conditional_render */

#ifdef GL_NV_conservative_raster

static GLboolean _glewInit_GL_NV_conservative_raster (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glSubpixelPrecisionBiasNV = (PFNGLSUBPIXELPRECISIONBIASNVPROC)glewGetProcAddress((const GLubyte*)"glSubpixelPrecisionBiasNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_conservative_raster */

#ifdef GL_NV_conservative_raster_dilate

static GLboolean _glewInit_GL_NV_conservative_raster_dilate (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glConservativeRasterParameterfNV = (PFNGLCONSERVATIVERASTERPARAMETERFNVPROC)glewGetProcAddress((const GLubyte*)"glConservativeRasterParameterfNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_conservative_raster_dilate */

#ifdef GL_NV_copy_image

static GLboolean _glewInit_GL_NV_copy_image (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCopyImageSubDataNV = (PFNGLCOPYIMAGESUBDATANVPROC)glewGetProcAddress((const GLubyte*)"glCopyImageSubDataNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_copy_image */

#ifdef GL_NV_depth_buffer_float

static GLboolean _glewInit_GL_NV_depth_buffer_float (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClearDepthdNV = (PFNGLCLEARDEPTHDNVPROC)glewGetProcAddress((const GLubyte*)"glClearDepthdNV")) == NULL) || r;
  r = ((glDepthBoundsdNV = (PFNGLDEPTHBOUNDSDNVPROC)glewGetProcAddress((const GLubyte*)"glDepthBoundsdNV")) == NULL) || r;
  r = ((glDepthRangedNV = (PFNGLDEPTHRANGEDNVPROC)glewGetProcAddress((const GLubyte*)"glDepthRangedNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_depth_buffer_float */

#ifdef GL_NV_draw_texture

static GLboolean _glewInit_GL_NV_draw_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDrawTextureNV = (PFNGLDRAWTEXTURENVPROC)glewGetProcAddress((const GLubyte*)"glDrawTextureNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_draw_texture */

#ifdef GL_NV_evaluators

static GLboolean _glewInit_GL_NV_evaluators (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glEvalMapsNV = (PFNGLEVALMAPSNVPROC)glewGetProcAddress((const GLubyte*)"glEvalMapsNV")) == NULL) || r;
  r = ((glGetMapAttribParameterfvNV = (PFNGLGETMAPATTRIBPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetMapAttribParameterfvNV")) == NULL) || r;
  r = ((glGetMapAttribParameterivNV = (PFNGLGETMAPATTRIBPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetMapAttribParameterivNV")) == NULL) || r;
  r = ((glGetMapControlPointsNV = (PFNGLGETMAPCONTROLPOINTSNVPROC)glewGetProcAddress((const GLubyte*)"glGetMapControlPointsNV")) == NULL) || r;
  r = ((glGetMapParameterfvNV = (PFNGLGETMAPPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetMapParameterfvNV")) == NULL) || r;
  r = ((glGetMapParameterivNV = (PFNGLGETMAPPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetMapParameterivNV")) == NULL) || r;
  r = ((glMapControlPointsNV = (PFNGLMAPCONTROLPOINTSNVPROC)glewGetProcAddress((const GLubyte*)"glMapControlPointsNV")) == NULL) || r;
  r = ((glMapParameterfvNV = (PFNGLMAPPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glMapParameterfvNV")) == NULL) || r;
  r = ((glMapParameterivNV = (PFNGLMAPPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glMapParameterivNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_evaluators */

#ifdef GL_NV_explicit_multisample

static GLboolean _glewInit_GL_NV_explicit_multisample (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetMultisamplefvNV = (PFNGLGETMULTISAMPLEFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetMultisamplefvNV")) == NULL) || r;
  r = ((glSampleMaskIndexedNV = (PFNGLSAMPLEMASKINDEXEDNVPROC)glewGetProcAddress((const GLubyte*)"glSampleMaskIndexedNV")) == NULL) || r;
  r = ((glTexRenderbufferNV = (PFNGLTEXRENDERBUFFERNVPROC)glewGetProcAddress((const GLubyte*)"glTexRenderbufferNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_explicit_multisample */

#ifdef GL_NV_fence

static GLboolean _glewInit_GL_NV_fence (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDeleteFencesNV = (PFNGLDELETEFENCESNVPROC)glewGetProcAddress((const GLubyte*)"glDeleteFencesNV")) == NULL) || r;
  r = ((glFinishFenceNV = (PFNGLFINISHFENCENVPROC)glewGetProcAddress((const GLubyte*)"glFinishFenceNV")) == NULL) || r;
  r = ((glGenFencesNV = (PFNGLGENFENCESNVPROC)glewGetProcAddress((const GLubyte*)"glGenFencesNV")) == NULL) || r;
  r = ((glGetFenceivNV = (PFNGLGETFENCEIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetFenceivNV")) == NULL) || r;
  r = ((glIsFenceNV = (PFNGLISFENCENVPROC)glewGetProcAddress((const GLubyte*)"glIsFenceNV")) == NULL) || r;
  r = ((glSetFenceNV = (PFNGLSETFENCENVPROC)glewGetProcAddress((const GLubyte*)"glSetFenceNV")) == NULL) || r;
  r = ((glTestFenceNV = (PFNGLTESTFENCENVPROC)glewGetProcAddress((const GLubyte*)"glTestFenceNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_fence */

#ifdef GL_NV_fragment_coverage_to_color

static GLboolean _glewInit_GL_NV_fragment_coverage_to_color (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFragmentCoverageColorNV = (PFNGLFRAGMENTCOVERAGECOLORNVPROC)glewGetProcAddress((const GLubyte*)"glFragmentCoverageColorNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_fragment_coverage_to_color */

#ifdef GL_NV_fragment_program

static GLboolean _glewInit_GL_NV_fragment_program (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetProgramNamedParameterdvNV = (PFNGLGETPROGRAMNAMEDPARAMETERDVNVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramNamedParameterdvNV")) == NULL) || r;
  r = ((glGetProgramNamedParameterfvNV = (PFNGLGETPROGRAMNAMEDPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramNamedParameterfvNV")) == NULL) || r;
  r = ((glProgramNamedParameter4dNV = (PFNGLPROGRAMNAMEDPARAMETER4DNVPROC)glewGetProcAddress((const GLubyte*)"glProgramNamedParameter4dNV")) == NULL) || r;
  r = ((glProgramNamedParameter4dvNV = (PFNGLPROGRAMNAMEDPARAMETER4DVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramNamedParameter4dvNV")) == NULL) || r;
  r = ((glProgramNamedParameter4fNV = (PFNGLPROGRAMNAMEDPARAMETER4FNVPROC)glewGetProcAddress((const GLubyte*)"glProgramNamedParameter4fNV")) == NULL) || r;
  r = ((glProgramNamedParameter4fvNV = (PFNGLPROGRAMNAMEDPARAMETER4FVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramNamedParameter4fvNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_fragment_program */

#ifdef GL_NV_framebuffer_multisample_coverage

static GLboolean _glewInit_GL_NV_framebuffer_multisample_coverage (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glRenderbufferStorageMultisampleCoverageNV = (PFNGLRENDERBUFFERSTORAGEMULTISAMPLECOVERAGENVPROC)glewGetProcAddress((const GLubyte*)"glRenderbufferStorageMultisampleCoverageNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_framebuffer_multisample_coverage */

#ifdef GL_NV_geometry_program4

static GLboolean _glewInit_GL_NV_geometry_program4 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glProgramVertexLimitNV = (PFNGLPROGRAMVERTEXLIMITNVPROC)glewGetProcAddress((const GLubyte*)"glProgramVertexLimitNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_geometry_program4 */

#ifdef GL_NV_gpu_program4

static GLboolean _glewInit_GL_NV_gpu_program4 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glProgramEnvParameterI4iNV = (PFNGLPROGRAMENVPARAMETERI4INVPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParameterI4iNV")) == NULL) || r;
  r = ((glProgramEnvParameterI4ivNV = (PFNGLPROGRAMENVPARAMETERI4IVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParameterI4ivNV")) == NULL) || r;
  r = ((glProgramEnvParameterI4uiNV = (PFNGLPROGRAMENVPARAMETERI4UINVPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParameterI4uiNV")) == NULL) || r;
  r = ((glProgramEnvParameterI4uivNV = (PFNGLPROGRAMENVPARAMETERI4UIVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParameterI4uivNV")) == NULL) || r;
  r = ((glProgramEnvParametersI4ivNV = (PFNGLPROGRAMENVPARAMETERSI4IVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParametersI4ivNV")) == NULL) || r;
  r = ((glProgramEnvParametersI4uivNV = (PFNGLPROGRAMENVPARAMETERSI4UIVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramEnvParametersI4uivNV")) == NULL) || r;
  r = ((glProgramLocalParameterI4iNV = (PFNGLPROGRAMLOCALPARAMETERI4INVPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParameterI4iNV")) == NULL) || r;
  r = ((glProgramLocalParameterI4ivNV = (PFNGLPROGRAMLOCALPARAMETERI4IVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParameterI4ivNV")) == NULL) || r;
  r = ((glProgramLocalParameterI4uiNV = (PFNGLPROGRAMLOCALPARAMETERI4UINVPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParameterI4uiNV")) == NULL) || r;
  r = ((glProgramLocalParameterI4uivNV = (PFNGLPROGRAMLOCALPARAMETERI4UIVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParameterI4uivNV")) == NULL) || r;
  r = ((glProgramLocalParametersI4ivNV = (PFNGLPROGRAMLOCALPARAMETERSI4IVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParametersI4ivNV")) == NULL) || r;
  r = ((glProgramLocalParametersI4uivNV = (PFNGLPROGRAMLOCALPARAMETERSI4UIVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramLocalParametersI4uivNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_gpu_program4 */

#ifdef GL_NV_gpu_shader5

static GLboolean _glewInit_GL_NV_gpu_shader5 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetUniformi64vNV = (PFNGLGETUNIFORMI64VNVPROC)glewGetProcAddress((const GLubyte*)"glGetUniformi64vNV")) == NULL) || r;
  r = ((glGetUniformui64vNV = (PFNGLGETUNIFORMUI64VNVPROC)glewGetProcAddress((const GLubyte*)"glGetUniformui64vNV")) == NULL) || r;
  r = ((glProgramUniform1i64NV = (PFNGLPROGRAMUNIFORM1I64NVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1i64NV")) == NULL) || r;
  r = ((glProgramUniform1i64vNV = (PFNGLPROGRAMUNIFORM1I64VNVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1i64vNV")) == NULL) || r;
  r = ((glProgramUniform1ui64NV = (PFNGLPROGRAMUNIFORM1UI64NVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1ui64NV")) == NULL) || r;
  r = ((glProgramUniform1ui64vNV = (PFNGLPROGRAMUNIFORM1UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform1ui64vNV")) == NULL) || r;
  r = ((glProgramUniform2i64NV = (PFNGLPROGRAMUNIFORM2I64NVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2i64NV")) == NULL) || r;
  r = ((glProgramUniform2i64vNV = (PFNGLPROGRAMUNIFORM2I64VNVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2i64vNV")) == NULL) || r;
  r = ((glProgramUniform2ui64NV = (PFNGLPROGRAMUNIFORM2UI64NVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2ui64NV")) == NULL) || r;
  r = ((glProgramUniform2ui64vNV = (PFNGLPROGRAMUNIFORM2UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform2ui64vNV")) == NULL) || r;
  r = ((glProgramUniform3i64NV = (PFNGLPROGRAMUNIFORM3I64NVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3i64NV")) == NULL) || r;
  r = ((glProgramUniform3i64vNV = (PFNGLPROGRAMUNIFORM3I64VNVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3i64vNV")) == NULL) || r;
  r = ((glProgramUniform3ui64NV = (PFNGLPROGRAMUNIFORM3UI64NVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3ui64NV")) == NULL) || r;
  r = ((glProgramUniform3ui64vNV = (PFNGLPROGRAMUNIFORM3UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform3ui64vNV")) == NULL) || r;
  r = ((glProgramUniform4i64NV = (PFNGLPROGRAMUNIFORM4I64NVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4i64NV")) == NULL) || r;
  r = ((glProgramUniform4i64vNV = (PFNGLPROGRAMUNIFORM4I64VNVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4i64vNV")) == NULL) || r;
  r = ((glProgramUniform4ui64NV = (PFNGLPROGRAMUNIFORM4UI64NVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4ui64NV")) == NULL) || r;
  r = ((glProgramUniform4ui64vNV = (PFNGLPROGRAMUNIFORM4UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniform4ui64vNV")) == NULL) || r;
  r = ((glUniform1i64NV = (PFNGLUNIFORM1I64NVPROC)glewGetProcAddress((const GLubyte*)"glUniform1i64NV")) == NULL) || r;
  r = ((glUniform1i64vNV = (PFNGLUNIFORM1I64VNVPROC)glewGetProcAddress((const GLubyte*)"glUniform1i64vNV")) == NULL) || r;
  r = ((glUniform1ui64NV = (PFNGLUNIFORM1UI64NVPROC)glewGetProcAddress((const GLubyte*)"glUniform1ui64NV")) == NULL) || r;
  r = ((glUniform1ui64vNV = (PFNGLUNIFORM1UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glUniform1ui64vNV")) == NULL) || r;
  r = ((glUniform2i64NV = (PFNGLUNIFORM2I64NVPROC)glewGetProcAddress((const GLubyte*)"glUniform2i64NV")) == NULL) || r;
  r = ((glUniform2i64vNV = (PFNGLUNIFORM2I64VNVPROC)glewGetProcAddress((const GLubyte*)"glUniform2i64vNV")) == NULL) || r;
  r = ((glUniform2ui64NV = (PFNGLUNIFORM2UI64NVPROC)glewGetProcAddress((const GLubyte*)"glUniform2ui64NV")) == NULL) || r;
  r = ((glUniform2ui64vNV = (PFNGLUNIFORM2UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glUniform2ui64vNV")) == NULL) || r;
  r = ((glUniform3i64NV = (PFNGLUNIFORM3I64NVPROC)glewGetProcAddress((const GLubyte*)"glUniform3i64NV")) == NULL) || r;
  r = ((glUniform3i64vNV = (PFNGLUNIFORM3I64VNVPROC)glewGetProcAddress((const GLubyte*)"glUniform3i64vNV")) == NULL) || r;
  r = ((glUniform3ui64NV = (PFNGLUNIFORM3UI64NVPROC)glewGetProcAddress((const GLubyte*)"glUniform3ui64NV")) == NULL) || r;
  r = ((glUniform3ui64vNV = (PFNGLUNIFORM3UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glUniform3ui64vNV")) == NULL) || r;
  r = ((glUniform4i64NV = (PFNGLUNIFORM4I64NVPROC)glewGetProcAddress((const GLubyte*)"glUniform4i64NV")) == NULL) || r;
  r = ((glUniform4i64vNV = (PFNGLUNIFORM4I64VNVPROC)glewGetProcAddress((const GLubyte*)"glUniform4i64vNV")) == NULL) || r;
  r = ((glUniform4ui64NV = (PFNGLUNIFORM4UI64NVPROC)glewGetProcAddress((const GLubyte*)"glUniform4ui64NV")) == NULL) || r;
  r = ((glUniform4ui64vNV = (PFNGLUNIFORM4UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glUniform4ui64vNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_gpu_shader5 */

#ifdef GL_NV_half_float

static GLboolean _glewInit_GL_NV_half_float (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glColor3hNV = (PFNGLCOLOR3HNVPROC)glewGetProcAddress((const GLubyte*)"glColor3hNV")) == NULL) || r;
  r = ((glColor3hvNV = (PFNGLCOLOR3HVNVPROC)glewGetProcAddress((const GLubyte*)"glColor3hvNV")) == NULL) || r;
  r = ((glColor4hNV = (PFNGLCOLOR4HNVPROC)glewGetProcAddress((const GLubyte*)"glColor4hNV")) == NULL) || r;
  r = ((glColor4hvNV = (PFNGLCOLOR4HVNVPROC)glewGetProcAddress((const GLubyte*)"glColor4hvNV")) == NULL) || r;
  r = ((glFogCoordhNV = (PFNGLFOGCOORDHNVPROC)glewGetProcAddress((const GLubyte*)"glFogCoordhNV")) == NULL) || r;
  r = ((glFogCoordhvNV = (PFNGLFOGCOORDHVNVPROC)glewGetProcAddress((const GLubyte*)"glFogCoordhvNV")) == NULL) || r;
  r = ((glMultiTexCoord1hNV = (PFNGLMULTITEXCOORD1HNVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1hNV")) == NULL) || r;
  r = ((glMultiTexCoord1hvNV = (PFNGLMULTITEXCOORD1HVNVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord1hvNV")) == NULL) || r;
  r = ((glMultiTexCoord2hNV = (PFNGLMULTITEXCOORD2HNVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2hNV")) == NULL) || r;
  r = ((glMultiTexCoord2hvNV = (PFNGLMULTITEXCOORD2HVNVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord2hvNV")) == NULL) || r;
  r = ((glMultiTexCoord3hNV = (PFNGLMULTITEXCOORD3HNVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3hNV")) == NULL) || r;
  r = ((glMultiTexCoord3hvNV = (PFNGLMULTITEXCOORD3HVNVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord3hvNV")) == NULL) || r;
  r = ((glMultiTexCoord4hNV = (PFNGLMULTITEXCOORD4HNVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4hNV")) == NULL) || r;
  r = ((glMultiTexCoord4hvNV = (PFNGLMULTITEXCOORD4HVNVPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4hvNV")) == NULL) || r;
  r = ((glNormal3hNV = (PFNGLNORMAL3HNVPROC)glewGetProcAddress((const GLubyte*)"glNormal3hNV")) == NULL) || r;
  r = ((glNormal3hvNV = (PFNGLNORMAL3HVNVPROC)glewGetProcAddress((const GLubyte*)"glNormal3hvNV")) == NULL) || r;
  r = ((glSecondaryColor3hNV = (PFNGLSECONDARYCOLOR3HNVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3hNV")) == NULL) || r;
  r = ((glSecondaryColor3hvNV = (PFNGLSECONDARYCOLOR3HVNVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColor3hvNV")) == NULL) || r;
  r = ((glTexCoord1hNV = (PFNGLTEXCOORD1HNVPROC)glewGetProcAddress((const GLubyte*)"glTexCoord1hNV")) == NULL) || r;
  r = ((glTexCoord1hvNV = (PFNGLTEXCOORD1HVNVPROC)glewGetProcAddress((const GLubyte*)"glTexCoord1hvNV")) == NULL) || r;
  r = ((glTexCoord2hNV = (PFNGLTEXCOORD2HNVPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2hNV")) == NULL) || r;
  r = ((glTexCoord2hvNV = (PFNGLTEXCOORD2HVNVPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2hvNV")) == NULL) || r;
  r = ((glTexCoord3hNV = (PFNGLTEXCOORD3HNVPROC)glewGetProcAddress((const GLubyte*)"glTexCoord3hNV")) == NULL) || r;
  r = ((glTexCoord3hvNV = (PFNGLTEXCOORD3HVNVPROC)glewGetProcAddress((const GLubyte*)"glTexCoord3hvNV")) == NULL) || r;
  r = ((glTexCoord4hNV = (PFNGLTEXCOORD4HNVPROC)glewGetProcAddress((const GLubyte*)"glTexCoord4hNV")) == NULL) || r;
  r = ((glTexCoord4hvNV = (PFNGLTEXCOORD4HVNVPROC)glewGetProcAddress((const GLubyte*)"glTexCoord4hvNV")) == NULL) || r;
  r = ((glVertex2hNV = (PFNGLVERTEX2HNVPROC)glewGetProcAddress((const GLubyte*)"glVertex2hNV")) == NULL) || r;
  r = ((glVertex2hvNV = (PFNGLVERTEX2HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertex2hvNV")) == NULL) || r;
  r = ((glVertex3hNV = (PFNGLVERTEX3HNVPROC)glewGetProcAddress((const GLubyte*)"glVertex3hNV")) == NULL) || r;
  r = ((glVertex3hvNV = (PFNGLVERTEX3HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertex3hvNV")) == NULL) || r;
  r = ((glVertex4hNV = (PFNGLVERTEX4HNVPROC)glewGetProcAddress((const GLubyte*)"glVertex4hNV")) == NULL) || r;
  r = ((glVertex4hvNV = (PFNGLVERTEX4HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertex4hvNV")) == NULL) || r;
  r = ((glVertexAttrib1hNV = (PFNGLVERTEXATTRIB1HNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1hNV")) == NULL) || r;
  r = ((glVertexAttrib1hvNV = (PFNGLVERTEXATTRIB1HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1hvNV")) == NULL) || r;
  r = ((glVertexAttrib2hNV = (PFNGLVERTEXATTRIB2HNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2hNV")) == NULL) || r;
  r = ((glVertexAttrib2hvNV = (PFNGLVERTEXATTRIB2HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2hvNV")) == NULL) || r;
  r = ((glVertexAttrib3hNV = (PFNGLVERTEXATTRIB3HNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3hNV")) == NULL) || r;
  r = ((glVertexAttrib3hvNV = (PFNGLVERTEXATTRIB3HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3hvNV")) == NULL) || r;
  r = ((glVertexAttrib4hNV = (PFNGLVERTEXATTRIB4HNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4hNV")) == NULL) || r;
  r = ((glVertexAttrib4hvNV = (PFNGLVERTEXATTRIB4HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4hvNV")) == NULL) || r;
  r = ((glVertexAttribs1hvNV = (PFNGLVERTEXATTRIBS1HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs1hvNV")) == NULL) || r;
  r = ((glVertexAttribs2hvNV = (PFNGLVERTEXATTRIBS2HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs2hvNV")) == NULL) || r;
  r = ((glVertexAttribs3hvNV = (PFNGLVERTEXATTRIBS3HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs3hvNV")) == NULL) || r;
  r = ((glVertexAttribs4hvNV = (PFNGLVERTEXATTRIBS4HVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs4hvNV")) == NULL) || r;
  r = ((glVertexWeighthNV = (PFNGLVERTEXWEIGHTHNVPROC)glewGetProcAddress((const GLubyte*)"glVertexWeighthNV")) == NULL) || r;
  r = ((glVertexWeighthvNV = (PFNGLVERTEXWEIGHTHVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexWeighthvNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_half_float */

#ifdef GL_NV_internalformat_sample_query

static GLboolean _glewInit_GL_NV_internalformat_sample_query (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetInternalformatSampleivNV = (PFNGLGETINTERNALFORMATSAMPLEIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetInternalformatSampleivNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_internalformat_sample_query */

#ifdef GL_NV_occlusion_query

static GLboolean _glewInit_GL_NV_occlusion_query (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginOcclusionQueryNV = (PFNGLBEGINOCCLUSIONQUERYNVPROC)glewGetProcAddress((const GLubyte*)"glBeginOcclusionQueryNV")) == NULL) || r;
  r = ((glDeleteOcclusionQueriesNV = (PFNGLDELETEOCCLUSIONQUERIESNVPROC)glewGetProcAddress((const GLubyte*)"glDeleteOcclusionQueriesNV")) == NULL) || r;
  r = ((glEndOcclusionQueryNV = (PFNGLENDOCCLUSIONQUERYNVPROC)glewGetProcAddress((const GLubyte*)"glEndOcclusionQueryNV")) == NULL) || r;
  r = ((glGenOcclusionQueriesNV = (PFNGLGENOCCLUSIONQUERIESNVPROC)glewGetProcAddress((const GLubyte*)"glGenOcclusionQueriesNV")) == NULL) || r;
  r = ((glGetOcclusionQueryivNV = (PFNGLGETOCCLUSIONQUERYIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetOcclusionQueryivNV")) == NULL) || r;
  r = ((glGetOcclusionQueryuivNV = (PFNGLGETOCCLUSIONQUERYUIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetOcclusionQueryuivNV")) == NULL) || r;
  r = ((glIsOcclusionQueryNV = (PFNGLISOCCLUSIONQUERYNVPROC)glewGetProcAddress((const GLubyte*)"glIsOcclusionQueryNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_occlusion_query */

#ifdef GL_NV_parameter_buffer_object

static GLboolean _glewInit_GL_NV_parameter_buffer_object (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glProgramBufferParametersIivNV = (PFNGLPROGRAMBUFFERPARAMETERSIIVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramBufferParametersIivNV")) == NULL) || r;
  r = ((glProgramBufferParametersIuivNV = (PFNGLPROGRAMBUFFERPARAMETERSIUIVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramBufferParametersIuivNV")) == NULL) || r;
  r = ((glProgramBufferParametersfvNV = (PFNGLPROGRAMBUFFERPARAMETERSFVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramBufferParametersfvNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_parameter_buffer_object */

#ifdef GL_NV_path_rendering

static GLboolean _glewInit_GL_NV_path_rendering (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCopyPathNV = (PFNGLCOPYPATHNVPROC)glewGetProcAddress((const GLubyte*)"glCopyPathNV")) == NULL) || r;
  r = ((glCoverFillPathInstancedNV = (PFNGLCOVERFILLPATHINSTANCEDNVPROC)glewGetProcAddress((const GLubyte*)"glCoverFillPathInstancedNV")) == NULL) || r;
  r = ((glCoverFillPathNV = (PFNGLCOVERFILLPATHNVPROC)glewGetProcAddress((const GLubyte*)"glCoverFillPathNV")) == NULL) || r;
  r = ((glCoverStrokePathInstancedNV = (PFNGLCOVERSTROKEPATHINSTANCEDNVPROC)glewGetProcAddress((const GLubyte*)"glCoverStrokePathInstancedNV")) == NULL) || r;
  r = ((glCoverStrokePathNV = (PFNGLCOVERSTROKEPATHNVPROC)glewGetProcAddress((const GLubyte*)"glCoverStrokePathNV")) == NULL) || r;
  r = ((glDeletePathsNV = (PFNGLDELETEPATHSNVPROC)glewGetProcAddress((const GLubyte*)"glDeletePathsNV")) == NULL) || r;
  r = ((glGenPathsNV = (PFNGLGENPATHSNVPROC)glewGetProcAddress((const GLubyte*)"glGenPathsNV")) == NULL) || r;
  r = ((glGetPathColorGenfvNV = (PFNGLGETPATHCOLORGENFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathColorGenfvNV")) == NULL) || r;
  r = ((glGetPathColorGenivNV = (PFNGLGETPATHCOLORGENIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathColorGenivNV")) == NULL) || r;
  r = ((glGetPathCommandsNV = (PFNGLGETPATHCOMMANDSNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathCommandsNV")) == NULL) || r;
  r = ((glGetPathCoordsNV = (PFNGLGETPATHCOORDSNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathCoordsNV")) == NULL) || r;
  r = ((glGetPathDashArrayNV = (PFNGLGETPATHDASHARRAYNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathDashArrayNV")) == NULL) || r;
  r = ((glGetPathLengthNV = (PFNGLGETPATHLENGTHNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathLengthNV")) == NULL) || r;
  r = ((glGetPathMetricRangeNV = (PFNGLGETPATHMETRICRANGENVPROC)glewGetProcAddress((const GLubyte*)"glGetPathMetricRangeNV")) == NULL) || r;
  r = ((glGetPathMetricsNV = (PFNGLGETPATHMETRICSNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathMetricsNV")) == NULL) || r;
  r = ((glGetPathParameterfvNV = (PFNGLGETPATHPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathParameterfvNV")) == NULL) || r;
  r = ((glGetPathParameterivNV = (PFNGLGETPATHPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathParameterivNV")) == NULL) || r;
  r = ((glGetPathSpacingNV = (PFNGLGETPATHSPACINGNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathSpacingNV")) == NULL) || r;
  r = ((glGetPathTexGenfvNV = (PFNGLGETPATHTEXGENFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathTexGenfvNV")) == NULL) || r;
  r = ((glGetPathTexGenivNV = (PFNGLGETPATHTEXGENIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetPathTexGenivNV")) == NULL) || r;
  r = ((glGetProgramResourcefvNV = (PFNGLGETPROGRAMRESOURCEFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramResourcefvNV")) == NULL) || r;
  r = ((glInterpolatePathsNV = (PFNGLINTERPOLATEPATHSNVPROC)glewGetProcAddress((const GLubyte*)"glInterpolatePathsNV")) == NULL) || r;
  r = ((glIsPathNV = (PFNGLISPATHNVPROC)glewGetProcAddress((const GLubyte*)"glIsPathNV")) == NULL) || r;
  r = ((glIsPointInFillPathNV = (PFNGLISPOINTINFILLPATHNVPROC)glewGetProcAddress((const GLubyte*)"glIsPointInFillPathNV")) == NULL) || r;
  r = ((glIsPointInStrokePathNV = (PFNGLISPOINTINSTROKEPATHNVPROC)glewGetProcAddress((const GLubyte*)"glIsPointInStrokePathNV")) == NULL) || r;
  r = ((glMatrixLoad3x2fNV = (PFNGLMATRIXLOAD3X2FNVPROC)glewGetProcAddress((const GLubyte*)"glMatrixLoad3x2fNV")) == NULL) || r;
  r = ((glMatrixLoad3x3fNV = (PFNGLMATRIXLOAD3X3FNVPROC)glewGetProcAddress((const GLubyte*)"glMatrixLoad3x3fNV")) == NULL) || r;
  r = ((glMatrixLoadTranspose3x3fNV = (PFNGLMATRIXLOADTRANSPOSE3X3FNVPROC)glewGetProcAddress((const GLubyte*)"glMatrixLoadTranspose3x3fNV")) == NULL) || r;
  r = ((glMatrixMult3x2fNV = (PFNGLMATRIXMULT3X2FNVPROC)glewGetProcAddress((const GLubyte*)"glMatrixMult3x2fNV")) == NULL) || r;
  r = ((glMatrixMult3x3fNV = (PFNGLMATRIXMULT3X3FNVPROC)glewGetProcAddress((const GLubyte*)"glMatrixMult3x3fNV")) == NULL) || r;
  r = ((glMatrixMultTranspose3x3fNV = (PFNGLMATRIXMULTTRANSPOSE3X3FNVPROC)glewGetProcAddress((const GLubyte*)"glMatrixMultTranspose3x3fNV")) == NULL) || r;
  r = ((glPathColorGenNV = (PFNGLPATHCOLORGENNVPROC)glewGetProcAddress((const GLubyte*)"glPathColorGenNV")) == NULL) || r;
  r = ((glPathCommandsNV = (PFNGLPATHCOMMANDSNVPROC)glewGetProcAddress((const GLubyte*)"glPathCommandsNV")) == NULL) || r;
  r = ((glPathCoordsNV = (PFNGLPATHCOORDSNVPROC)glewGetProcAddress((const GLubyte*)"glPathCoordsNV")) == NULL) || r;
  r = ((glPathCoverDepthFuncNV = (PFNGLPATHCOVERDEPTHFUNCNVPROC)glewGetProcAddress((const GLubyte*)"glPathCoverDepthFuncNV")) == NULL) || r;
  r = ((glPathDashArrayNV = (PFNGLPATHDASHARRAYNVPROC)glewGetProcAddress((const GLubyte*)"glPathDashArrayNV")) == NULL) || r;
  r = ((glPathFogGenNV = (PFNGLPATHFOGGENNVPROC)glewGetProcAddress((const GLubyte*)"glPathFogGenNV")) == NULL) || r;
  r = ((glPathGlyphIndexArrayNV = (PFNGLPATHGLYPHINDEXARRAYNVPROC)glewGetProcAddress((const GLubyte*)"glPathGlyphIndexArrayNV")) == NULL) || r;
  r = ((glPathGlyphIndexRangeNV = (PFNGLPATHGLYPHINDEXRANGENVPROC)glewGetProcAddress((const GLubyte*)"glPathGlyphIndexRangeNV")) == NULL) || r;
  r = ((glPathGlyphRangeNV = (PFNGLPATHGLYPHRANGENVPROC)glewGetProcAddress((const GLubyte*)"glPathGlyphRangeNV")) == NULL) || r;
  r = ((glPathGlyphsNV = (PFNGLPATHGLYPHSNVPROC)glewGetProcAddress((const GLubyte*)"glPathGlyphsNV")) == NULL) || r;
  r = ((glPathMemoryGlyphIndexArrayNV = (PFNGLPATHMEMORYGLYPHINDEXARRAYNVPROC)glewGetProcAddress((const GLubyte*)"glPathMemoryGlyphIndexArrayNV")) == NULL) || r;
  r = ((glPathParameterfNV = (PFNGLPATHPARAMETERFNVPROC)glewGetProcAddress((const GLubyte*)"glPathParameterfNV")) == NULL) || r;
  r = ((glPathParameterfvNV = (PFNGLPATHPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glPathParameterfvNV")) == NULL) || r;
  r = ((glPathParameteriNV = (PFNGLPATHPARAMETERINVPROC)glewGetProcAddress((const GLubyte*)"glPathParameteriNV")) == NULL) || r;
  r = ((glPathParameterivNV = (PFNGLPATHPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glPathParameterivNV")) == NULL) || r;
  r = ((glPathStencilDepthOffsetNV = (PFNGLPATHSTENCILDEPTHOFFSETNVPROC)glewGetProcAddress((const GLubyte*)"glPathStencilDepthOffsetNV")) == NULL) || r;
  r = ((glPathStencilFuncNV = (PFNGLPATHSTENCILFUNCNVPROC)glewGetProcAddress((const GLubyte*)"glPathStencilFuncNV")) == NULL) || r;
  r = ((glPathStringNV = (PFNGLPATHSTRINGNVPROC)glewGetProcAddress((const GLubyte*)"glPathStringNV")) == NULL) || r;
  r = ((glPathSubCommandsNV = (PFNGLPATHSUBCOMMANDSNVPROC)glewGetProcAddress((const GLubyte*)"glPathSubCommandsNV")) == NULL) || r;
  r = ((glPathSubCoordsNV = (PFNGLPATHSUBCOORDSNVPROC)glewGetProcAddress((const GLubyte*)"glPathSubCoordsNV")) == NULL) || r;
  r = ((glPathTexGenNV = (PFNGLPATHTEXGENNVPROC)glewGetProcAddress((const GLubyte*)"glPathTexGenNV")) == NULL) || r;
  r = ((glPointAlongPathNV = (PFNGLPOINTALONGPATHNVPROC)glewGetProcAddress((const GLubyte*)"glPointAlongPathNV")) == NULL) || r;
  r = ((glProgramPathFragmentInputGenNV = (PFNGLPROGRAMPATHFRAGMENTINPUTGENNVPROC)glewGetProcAddress((const GLubyte*)"glProgramPathFragmentInputGenNV")) == NULL) || r;
  r = ((glStencilFillPathInstancedNV = (PFNGLSTENCILFILLPATHINSTANCEDNVPROC)glewGetProcAddress((const GLubyte*)"glStencilFillPathInstancedNV")) == NULL) || r;
  r = ((glStencilFillPathNV = (PFNGLSTENCILFILLPATHNVPROC)glewGetProcAddress((const GLubyte*)"glStencilFillPathNV")) == NULL) || r;
  r = ((glStencilStrokePathInstancedNV = (PFNGLSTENCILSTROKEPATHINSTANCEDNVPROC)glewGetProcAddress((const GLubyte*)"glStencilStrokePathInstancedNV")) == NULL) || r;
  r = ((glStencilStrokePathNV = (PFNGLSTENCILSTROKEPATHNVPROC)glewGetProcAddress((const GLubyte*)"glStencilStrokePathNV")) == NULL) || r;
  r = ((glStencilThenCoverFillPathInstancedNV = (PFNGLSTENCILTHENCOVERFILLPATHINSTANCEDNVPROC)glewGetProcAddress((const GLubyte*)"glStencilThenCoverFillPathInstancedNV")) == NULL) || r;
  r = ((glStencilThenCoverFillPathNV = (PFNGLSTENCILTHENCOVERFILLPATHNVPROC)glewGetProcAddress((const GLubyte*)"glStencilThenCoverFillPathNV")) == NULL) || r;
  r = ((glStencilThenCoverStrokePathInstancedNV = (PFNGLSTENCILTHENCOVERSTROKEPATHINSTANCEDNVPROC)glewGetProcAddress((const GLubyte*)"glStencilThenCoverStrokePathInstancedNV")) == NULL) || r;
  r = ((glStencilThenCoverStrokePathNV = (PFNGLSTENCILTHENCOVERSTROKEPATHNVPROC)glewGetProcAddress((const GLubyte*)"glStencilThenCoverStrokePathNV")) == NULL) || r;
  r = ((glTransformPathNV = (PFNGLTRANSFORMPATHNVPROC)glewGetProcAddress((const GLubyte*)"glTransformPathNV")) == NULL) || r;
  r = ((glWeightPathsNV = (PFNGLWEIGHTPATHSNVPROC)glewGetProcAddress((const GLubyte*)"glWeightPathsNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_path_rendering */

#ifdef GL_NV_pixel_data_range

static GLboolean _glewInit_GL_NV_pixel_data_range (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFlushPixelDataRangeNV = (PFNGLFLUSHPIXELDATARANGENVPROC)glewGetProcAddress((const GLubyte*)"glFlushPixelDataRangeNV")) == NULL) || r;
  r = ((glPixelDataRangeNV = (PFNGLPIXELDATARANGENVPROC)glewGetProcAddress((const GLubyte*)"glPixelDataRangeNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_pixel_data_range */

#ifdef GL_NV_point_sprite

static GLboolean _glewInit_GL_NV_point_sprite (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glPointParameteriNV = (PFNGLPOINTPARAMETERINVPROC)glewGetProcAddress((const GLubyte*)"glPointParameteriNV")) == NULL) || r;
  r = ((glPointParameterivNV = (PFNGLPOINTPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glPointParameterivNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_point_sprite */

#ifdef GL_NV_present_video

static GLboolean _glewInit_GL_NV_present_video (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetVideoi64vNV = (PFNGLGETVIDEOI64VNVPROC)glewGetProcAddress((const GLubyte*)"glGetVideoi64vNV")) == NULL) || r;
  r = ((glGetVideoivNV = (PFNGLGETVIDEOIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetVideoivNV")) == NULL) || r;
  r = ((glGetVideoui64vNV = (PFNGLGETVIDEOUI64VNVPROC)glewGetProcAddress((const GLubyte*)"glGetVideoui64vNV")) == NULL) || r;
  r = ((glGetVideouivNV = (PFNGLGETVIDEOUIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetVideouivNV")) == NULL) || r;
  r = ((glPresentFrameDualFillNV = (PFNGLPRESENTFRAMEDUALFILLNVPROC)glewGetProcAddress((const GLubyte*)"glPresentFrameDualFillNV")) == NULL) || r;
  r = ((glPresentFrameKeyedNV = (PFNGLPRESENTFRAMEKEYEDNVPROC)glewGetProcAddress((const GLubyte*)"glPresentFrameKeyedNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_present_video */

#ifdef GL_NV_primitive_restart

static GLboolean _glewInit_GL_NV_primitive_restart (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glPrimitiveRestartIndexNV = (PFNGLPRIMITIVERESTARTINDEXNVPROC)glewGetProcAddress((const GLubyte*)"glPrimitiveRestartIndexNV")) == NULL) || r;
  r = ((glPrimitiveRestartNV = (PFNGLPRIMITIVERESTARTNVPROC)glewGetProcAddress((const GLubyte*)"glPrimitiveRestartNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_primitive_restart */

#ifdef GL_NV_register_combiners

static GLboolean _glewInit_GL_NV_register_combiners (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCombinerInputNV = (PFNGLCOMBINERINPUTNVPROC)glewGetProcAddress((const GLubyte*)"glCombinerInputNV")) == NULL) || r;
  r = ((glCombinerOutputNV = (PFNGLCOMBINEROUTPUTNVPROC)glewGetProcAddress((const GLubyte*)"glCombinerOutputNV")) == NULL) || r;
  r = ((glCombinerParameterfNV = (PFNGLCOMBINERPARAMETERFNVPROC)glewGetProcAddress((const GLubyte*)"glCombinerParameterfNV")) == NULL) || r;
  r = ((glCombinerParameterfvNV = (PFNGLCOMBINERPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glCombinerParameterfvNV")) == NULL) || r;
  r = ((glCombinerParameteriNV = (PFNGLCOMBINERPARAMETERINVPROC)glewGetProcAddress((const GLubyte*)"glCombinerParameteriNV")) == NULL) || r;
  r = ((glCombinerParameterivNV = (PFNGLCOMBINERPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glCombinerParameterivNV")) == NULL) || r;
  r = ((glFinalCombinerInputNV = (PFNGLFINALCOMBINERINPUTNVPROC)glewGetProcAddress((const GLubyte*)"glFinalCombinerInputNV")) == NULL) || r;
  r = ((glGetCombinerInputParameterfvNV = (PFNGLGETCOMBINERINPUTPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetCombinerInputParameterfvNV")) == NULL) || r;
  r = ((glGetCombinerInputParameterivNV = (PFNGLGETCOMBINERINPUTPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetCombinerInputParameterivNV")) == NULL) || r;
  r = ((glGetCombinerOutputParameterfvNV = (PFNGLGETCOMBINEROUTPUTPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetCombinerOutputParameterfvNV")) == NULL) || r;
  r = ((glGetCombinerOutputParameterivNV = (PFNGLGETCOMBINEROUTPUTPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetCombinerOutputParameterivNV")) == NULL) || r;
  r = ((glGetFinalCombinerInputParameterfvNV = (PFNGLGETFINALCOMBINERINPUTPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetFinalCombinerInputParameterfvNV")) == NULL) || r;
  r = ((glGetFinalCombinerInputParameterivNV = (PFNGLGETFINALCOMBINERINPUTPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetFinalCombinerInputParameterivNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_register_combiners */

#ifdef GL_NV_register_combiners2

static GLboolean _glewInit_GL_NV_register_combiners2 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glCombinerStageParameterfvNV = (PFNGLCOMBINERSTAGEPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glCombinerStageParameterfvNV")) == NULL) || r;
  r = ((glGetCombinerStageParameterfvNV = (PFNGLGETCOMBINERSTAGEPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetCombinerStageParameterfvNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_register_combiners2 */

#ifdef GL_NV_sample_locations

static GLboolean _glewInit_GL_NV_sample_locations (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFramebufferSampleLocationsfvNV = (PFNGLFRAMEBUFFERSAMPLELOCATIONSFVNVPROC)glewGetProcAddress((const GLubyte*)"glFramebufferSampleLocationsfvNV")) == NULL) || r;
  r = ((glNamedFramebufferSampleLocationsfvNV = (PFNGLNAMEDFRAMEBUFFERSAMPLELOCATIONSFVNVPROC)glewGetProcAddress((const GLubyte*)"glNamedFramebufferSampleLocationsfvNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_sample_locations */

#ifdef GL_NV_shader_buffer_load

static GLboolean _glewInit_GL_NV_shader_buffer_load (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetBufferParameterui64vNV = (PFNGLGETBUFFERPARAMETERUI64VNVPROC)glewGetProcAddress((const GLubyte*)"glGetBufferParameterui64vNV")) == NULL) || r;
  r = ((glGetIntegerui64vNV = (PFNGLGETINTEGERUI64VNVPROC)glewGetProcAddress((const GLubyte*)"glGetIntegerui64vNV")) == NULL) || r;
  r = ((glGetNamedBufferParameterui64vNV = (PFNGLGETNAMEDBUFFERPARAMETERUI64VNVPROC)glewGetProcAddress((const GLubyte*)"glGetNamedBufferParameterui64vNV")) == NULL) || r;
  r = ((glIsBufferResidentNV = (PFNGLISBUFFERRESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glIsBufferResidentNV")) == NULL) || r;
  r = ((glIsNamedBufferResidentNV = (PFNGLISNAMEDBUFFERRESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glIsNamedBufferResidentNV")) == NULL) || r;
  r = ((glMakeBufferNonResidentNV = (PFNGLMAKEBUFFERNONRESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glMakeBufferNonResidentNV")) == NULL) || r;
  r = ((glMakeBufferResidentNV = (PFNGLMAKEBUFFERRESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glMakeBufferResidentNV")) == NULL) || r;
  r = ((glMakeNamedBufferNonResidentNV = (PFNGLMAKENAMEDBUFFERNONRESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glMakeNamedBufferNonResidentNV")) == NULL) || r;
  r = ((glMakeNamedBufferResidentNV = (PFNGLMAKENAMEDBUFFERRESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glMakeNamedBufferResidentNV")) == NULL) || r;
  r = ((glProgramUniformui64NV = (PFNGLPROGRAMUNIFORMUI64NVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformui64NV")) == NULL) || r;
  r = ((glProgramUniformui64vNV = (PFNGLPROGRAMUNIFORMUI64VNVPROC)glewGetProcAddress((const GLubyte*)"glProgramUniformui64vNV")) == NULL) || r;
  r = ((glUniformui64NV = (PFNGLUNIFORMUI64NVPROC)glewGetProcAddress((const GLubyte*)"glUniformui64NV")) == NULL) || r;
  r = ((glUniformui64vNV = (PFNGLUNIFORMUI64VNVPROC)glewGetProcAddress((const GLubyte*)"glUniformui64vNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_shader_buffer_load */

#ifdef GL_NV_texture_barrier

static GLboolean _glewInit_GL_NV_texture_barrier (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTextureBarrierNV = (PFNGLTEXTUREBARRIERNVPROC)glewGetProcAddress((const GLubyte*)"glTextureBarrierNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_texture_barrier */

#ifdef GL_NV_texture_multisample

static GLboolean _glewInit_GL_NV_texture_multisample (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexImage2DMultisampleCoverageNV = (PFNGLTEXIMAGE2DMULTISAMPLECOVERAGENVPROC)glewGetProcAddress((const GLubyte*)"glTexImage2DMultisampleCoverageNV")) == NULL) || r;
  r = ((glTexImage3DMultisampleCoverageNV = (PFNGLTEXIMAGE3DMULTISAMPLECOVERAGENVPROC)glewGetProcAddress((const GLubyte*)"glTexImage3DMultisampleCoverageNV")) == NULL) || r;
  r = ((glTextureImage2DMultisampleCoverageNV = (PFNGLTEXTUREIMAGE2DMULTISAMPLECOVERAGENVPROC)glewGetProcAddress((const GLubyte*)"glTextureImage2DMultisampleCoverageNV")) == NULL) || r;
  r = ((glTextureImage2DMultisampleNV = (PFNGLTEXTUREIMAGE2DMULTISAMPLENVPROC)glewGetProcAddress((const GLubyte*)"glTextureImage2DMultisampleNV")) == NULL) || r;
  r = ((glTextureImage3DMultisampleCoverageNV = (PFNGLTEXTUREIMAGE3DMULTISAMPLECOVERAGENVPROC)glewGetProcAddress((const GLubyte*)"glTextureImage3DMultisampleCoverageNV")) == NULL) || r;
  r = ((glTextureImage3DMultisampleNV = (PFNGLTEXTUREIMAGE3DMULTISAMPLENVPROC)glewGetProcAddress((const GLubyte*)"glTextureImage3DMultisampleNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_texture_multisample */

#ifdef GL_NV_transform_feedback

static GLboolean _glewInit_GL_NV_transform_feedback (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glActiveVaryingNV = (PFNGLACTIVEVARYINGNVPROC)glewGetProcAddress((const GLubyte*)"glActiveVaryingNV")) == NULL) || r;
  r = ((glBeginTransformFeedbackNV = (PFNGLBEGINTRANSFORMFEEDBACKNVPROC)glewGetProcAddress((const GLubyte*)"glBeginTransformFeedbackNV")) == NULL) || r;
  r = ((glBindBufferBaseNV = (PFNGLBINDBUFFERBASENVPROC)glewGetProcAddress((const GLubyte*)"glBindBufferBaseNV")) == NULL) || r;
  r = ((glBindBufferOffsetNV = (PFNGLBINDBUFFEROFFSETNVPROC)glewGetProcAddress((const GLubyte*)"glBindBufferOffsetNV")) == NULL) || r;
  r = ((glBindBufferRangeNV = (PFNGLBINDBUFFERRANGENVPROC)glewGetProcAddress((const GLubyte*)"glBindBufferRangeNV")) == NULL) || r;
  r = ((glEndTransformFeedbackNV = (PFNGLENDTRANSFORMFEEDBACKNVPROC)glewGetProcAddress((const GLubyte*)"glEndTransformFeedbackNV")) == NULL) || r;
  r = ((glGetActiveVaryingNV = (PFNGLGETACTIVEVARYINGNVPROC)glewGetProcAddress((const GLubyte*)"glGetActiveVaryingNV")) == NULL) || r;
  r = ((glGetTransformFeedbackVaryingNV = (PFNGLGETTRANSFORMFEEDBACKVARYINGNVPROC)glewGetProcAddress((const GLubyte*)"glGetTransformFeedbackVaryingNV")) == NULL) || r;
  r = ((glGetVaryingLocationNV = (PFNGLGETVARYINGLOCATIONNVPROC)glewGetProcAddress((const GLubyte*)"glGetVaryingLocationNV")) == NULL) || r;
  r = ((glTransformFeedbackAttribsNV = (PFNGLTRANSFORMFEEDBACKATTRIBSNVPROC)glewGetProcAddress((const GLubyte*)"glTransformFeedbackAttribsNV")) == NULL) || r;
  r = ((glTransformFeedbackVaryingsNV = (PFNGLTRANSFORMFEEDBACKVARYINGSNVPROC)glewGetProcAddress((const GLubyte*)"glTransformFeedbackVaryingsNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_transform_feedback */

#ifdef GL_NV_transform_feedback2

static GLboolean _glewInit_GL_NV_transform_feedback2 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBindTransformFeedbackNV = (PFNGLBINDTRANSFORMFEEDBACKNVPROC)glewGetProcAddress((const GLubyte*)"glBindTransformFeedbackNV")) == NULL) || r;
  r = ((glDeleteTransformFeedbacksNV = (PFNGLDELETETRANSFORMFEEDBACKSNVPROC)glewGetProcAddress((const GLubyte*)"glDeleteTransformFeedbacksNV")) == NULL) || r;
  r = ((glDrawTransformFeedbackNV = (PFNGLDRAWTRANSFORMFEEDBACKNVPROC)glewGetProcAddress((const GLubyte*)"glDrawTransformFeedbackNV")) == NULL) || r;
  r = ((glGenTransformFeedbacksNV = (PFNGLGENTRANSFORMFEEDBACKSNVPROC)glewGetProcAddress((const GLubyte*)"glGenTransformFeedbacksNV")) == NULL) || r;
  r = ((glIsTransformFeedbackNV = (PFNGLISTRANSFORMFEEDBACKNVPROC)glewGetProcAddress((const GLubyte*)"glIsTransformFeedbackNV")) == NULL) || r;
  r = ((glPauseTransformFeedbackNV = (PFNGLPAUSETRANSFORMFEEDBACKNVPROC)glewGetProcAddress((const GLubyte*)"glPauseTransformFeedbackNV")) == NULL) || r;
  r = ((glResumeTransformFeedbackNV = (PFNGLRESUMETRANSFORMFEEDBACKNVPROC)glewGetProcAddress((const GLubyte*)"glResumeTransformFeedbackNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_transform_feedback2 */

#ifdef GL_NV_vdpau_interop

static GLboolean _glewInit_GL_NV_vdpau_interop (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glVDPAUFiniNV = (PFNGLVDPAUFININVPROC)glewGetProcAddress((const GLubyte*)"glVDPAUFiniNV")) == NULL) || r;
  r = ((glVDPAUGetSurfaceivNV = (PFNGLVDPAUGETSURFACEIVNVPROC)glewGetProcAddress((const GLubyte*)"glVDPAUGetSurfaceivNV")) == NULL) || r;
  r = ((glVDPAUInitNV = (PFNGLVDPAUINITNVPROC)glewGetProcAddress((const GLubyte*)"glVDPAUInitNV")) == NULL) || r;
  r = ((glVDPAUIsSurfaceNV = (PFNGLVDPAUISSURFACENVPROC)glewGetProcAddress((const GLubyte*)"glVDPAUIsSurfaceNV")) == NULL) || r;
  r = ((glVDPAUMapSurfacesNV = (PFNGLVDPAUMAPSURFACESNVPROC)glewGetProcAddress((const GLubyte*)"glVDPAUMapSurfacesNV")) == NULL) || r;
  r = ((glVDPAURegisterOutputSurfaceNV = (PFNGLVDPAUREGISTEROUTPUTSURFACENVPROC)glewGetProcAddress((const GLubyte*)"glVDPAURegisterOutputSurfaceNV")) == NULL) || r;
  r = ((glVDPAURegisterVideoSurfaceNV = (PFNGLVDPAUREGISTERVIDEOSURFACENVPROC)glewGetProcAddress((const GLubyte*)"glVDPAURegisterVideoSurfaceNV")) == NULL) || r;
  r = ((glVDPAUSurfaceAccessNV = (PFNGLVDPAUSURFACEACCESSNVPROC)glewGetProcAddress((const GLubyte*)"glVDPAUSurfaceAccessNV")) == NULL) || r;
  r = ((glVDPAUUnmapSurfacesNV = (PFNGLVDPAUUNMAPSURFACESNVPROC)glewGetProcAddress((const GLubyte*)"glVDPAUUnmapSurfacesNV")) == NULL) || r;
  r = ((glVDPAUUnregisterSurfaceNV = (PFNGLVDPAUUNREGISTERSURFACENVPROC)glewGetProcAddress((const GLubyte*)"glVDPAUUnregisterSurfaceNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_vdpau_interop */

#ifdef GL_NV_vertex_array_range

static GLboolean _glewInit_GL_NV_vertex_array_range (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFlushVertexArrayRangeNV = (PFNGLFLUSHVERTEXARRAYRANGENVPROC)glewGetProcAddress((const GLubyte*)"glFlushVertexArrayRangeNV")) == NULL) || r;
  r = ((glVertexArrayRangeNV = (PFNGLVERTEXARRAYRANGENVPROC)glewGetProcAddress((const GLubyte*)"glVertexArrayRangeNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_vertex_array_range */

#ifdef GL_NV_vertex_attrib_integer_64bit

static GLboolean _glewInit_GL_NV_vertex_attrib_integer_64bit (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetVertexAttribLi64vNV = (PFNGLGETVERTEXATTRIBLI64VNVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribLi64vNV")) == NULL) || r;
  r = ((glGetVertexAttribLui64vNV = (PFNGLGETVERTEXATTRIBLUI64VNVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribLui64vNV")) == NULL) || r;
  r = ((glVertexAttribL1i64NV = (PFNGLVERTEXATTRIBL1I64NVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL1i64NV")) == NULL) || r;
  r = ((glVertexAttribL1i64vNV = (PFNGLVERTEXATTRIBL1I64VNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL1i64vNV")) == NULL) || r;
  r = ((glVertexAttribL1ui64NV = (PFNGLVERTEXATTRIBL1UI64NVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL1ui64NV")) == NULL) || r;
  r = ((glVertexAttribL1ui64vNV = (PFNGLVERTEXATTRIBL1UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL1ui64vNV")) == NULL) || r;
  r = ((glVertexAttribL2i64NV = (PFNGLVERTEXATTRIBL2I64NVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL2i64NV")) == NULL) || r;
  r = ((glVertexAttribL2i64vNV = (PFNGLVERTEXATTRIBL2I64VNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL2i64vNV")) == NULL) || r;
  r = ((glVertexAttribL2ui64NV = (PFNGLVERTEXATTRIBL2UI64NVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL2ui64NV")) == NULL) || r;
  r = ((glVertexAttribL2ui64vNV = (PFNGLVERTEXATTRIBL2UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL2ui64vNV")) == NULL) || r;
  r = ((glVertexAttribL3i64NV = (PFNGLVERTEXATTRIBL3I64NVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL3i64NV")) == NULL) || r;
  r = ((glVertexAttribL3i64vNV = (PFNGLVERTEXATTRIBL3I64VNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL3i64vNV")) == NULL) || r;
  r = ((glVertexAttribL3ui64NV = (PFNGLVERTEXATTRIBL3UI64NVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL3ui64NV")) == NULL) || r;
  r = ((glVertexAttribL3ui64vNV = (PFNGLVERTEXATTRIBL3UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL3ui64vNV")) == NULL) || r;
  r = ((glVertexAttribL4i64NV = (PFNGLVERTEXATTRIBL4I64NVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL4i64NV")) == NULL) || r;
  r = ((glVertexAttribL4i64vNV = (PFNGLVERTEXATTRIBL4I64VNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL4i64vNV")) == NULL) || r;
  r = ((glVertexAttribL4ui64NV = (PFNGLVERTEXATTRIBL4UI64NVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL4ui64NV")) == NULL) || r;
  r = ((glVertexAttribL4ui64vNV = (PFNGLVERTEXATTRIBL4UI64VNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribL4ui64vNV")) == NULL) || r;
  r = ((glVertexAttribLFormatNV = (PFNGLVERTEXATTRIBLFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribLFormatNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_vertex_attrib_integer_64bit */

#ifdef GL_NV_vertex_buffer_unified_memory

static GLboolean _glewInit_GL_NV_vertex_buffer_unified_memory (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBufferAddressRangeNV = (PFNGLBUFFERADDRESSRANGENVPROC)glewGetProcAddress((const GLubyte*)"glBufferAddressRangeNV")) == NULL) || r;
  r = ((glColorFormatNV = (PFNGLCOLORFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glColorFormatNV")) == NULL) || r;
  r = ((glEdgeFlagFormatNV = (PFNGLEDGEFLAGFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glEdgeFlagFormatNV")) == NULL) || r;
  r = ((glFogCoordFormatNV = (PFNGLFOGCOORDFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glFogCoordFormatNV")) == NULL) || r;
  r = ((glGetIntegerui64i_vNV = (PFNGLGETINTEGERUI64I_VNVPROC)glewGetProcAddress((const GLubyte*)"glGetIntegerui64i_vNV")) == NULL) || r;
  r = ((glIndexFormatNV = (PFNGLINDEXFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glIndexFormatNV")) == NULL) || r;
  r = ((glNormalFormatNV = (PFNGLNORMALFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glNormalFormatNV")) == NULL) || r;
  r = ((glSecondaryColorFormatNV = (PFNGLSECONDARYCOLORFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glSecondaryColorFormatNV")) == NULL) || r;
  r = ((glTexCoordFormatNV = (PFNGLTEXCOORDFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glTexCoordFormatNV")) == NULL) || r;
  r = ((glVertexAttribFormatNV = (PFNGLVERTEXATTRIBFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribFormatNV")) == NULL) || r;
  r = ((glVertexAttribIFormatNV = (PFNGLVERTEXATTRIBIFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribIFormatNV")) == NULL) || r;
  r = ((glVertexFormatNV = (PFNGLVERTEXFORMATNVPROC)glewGetProcAddress((const GLubyte*)"glVertexFormatNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_vertex_buffer_unified_memory */

#ifdef GL_NV_vertex_program

static GLboolean _glewInit_GL_NV_vertex_program (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glAreProgramsResidentNV = (PFNGLAREPROGRAMSRESIDENTNVPROC)glewGetProcAddress((const GLubyte*)"glAreProgramsResidentNV")) == NULL) || r;
  r = ((glBindProgramNV = (PFNGLBINDPROGRAMNVPROC)glewGetProcAddress((const GLubyte*)"glBindProgramNV")) == NULL) || r;
  r = ((glDeleteProgramsNV = (PFNGLDELETEPROGRAMSNVPROC)glewGetProcAddress((const GLubyte*)"glDeleteProgramsNV")) == NULL) || r;
  r = ((glExecuteProgramNV = (PFNGLEXECUTEPROGRAMNVPROC)glewGetProcAddress((const GLubyte*)"glExecuteProgramNV")) == NULL) || r;
  r = ((glGenProgramsNV = (PFNGLGENPROGRAMSNVPROC)glewGetProcAddress((const GLubyte*)"glGenProgramsNV")) == NULL) || r;
  r = ((glGetProgramParameterdvNV = (PFNGLGETPROGRAMPARAMETERDVNVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramParameterdvNV")) == NULL) || r;
  r = ((glGetProgramParameterfvNV = (PFNGLGETPROGRAMPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramParameterfvNV")) == NULL) || r;
  r = ((glGetProgramStringNV = (PFNGLGETPROGRAMSTRINGNVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramStringNV")) == NULL) || r;
  r = ((glGetProgramivNV = (PFNGLGETPROGRAMIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetProgramivNV")) == NULL) || r;
  r = ((glGetTrackMatrixivNV = (PFNGLGETTRACKMATRIXIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetTrackMatrixivNV")) == NULL) || r;
  r = ((glGetVertexAttribPointervNV = (PFNGLGETVERTEXATTRIBPOINTERVNVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribPointervNV")) == NULL) || r;
  r = ((glGetVertexAttribdvNV = (PFNGLGETVERTEXATTRIBDVNVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribdvNV")) == NULL) || r;
  r = ((glGetVertexAttribfvNV = (PFNGLGETVERTEXATTRIBFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribfvNV")) == NULL) || r;
  r = ((glGetVertexAttribivNV = (PFNGLGETVERTEXATTRIBIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetVertexAttribivNV")) == NULL) || r;
  r = ((glIsProgramNV = (PFNGLISPROGRAMNVPROC)glewGetProcAddress((const GLubyte*)"glIsProgramNV")) == NULL) || r;
  r = ((glLoadProgramNV = (PFNGLLOADPROGRAMNVPROC)glewGetProcAddress((const GLubyte*)"glLoadProgramNV")) == NULL) || r;
  r = ((glProgramParameter4dNV = (PFNGLPROGRAMPARAMETER4DNVPROC)glewGetProcAddress((const GLubyte*)"glProgramParameter4dNV")) == NULL) || r;
  r = ((glProgramParameter4dvNV = (PFNGLPROGRAMPARAMETER4DVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramParameter4dvNV")) == NULL) || r;
  r = ((glProgramParameter4fNV = (PFNGLPROGRAMPARAMETER4FNVPROC)glewGetProcAddress((const GLubyte*)"glProgramParameter4fNV")) == NULL) || r;
  r = ((glProgramParameter4fvNV = (PFNGLPROGRAMPARAMETER4FVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramParameter4fvNV")) == NULL) || r;
  r = ((glProgramParameters4dvNV = (PFNGLPROGRAMPARAMETERS4DVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramParameters4dvNV")) == NULL) || r;
  r = ((glProgramParameters4fvNV = (PFNGLPROGRAMPARAMETERS4FVNVPROC)glewGetProcAddress((const GLubyte*)"glProgramParameters4fvNV")) == NULL) || r;
  r = ((glRequestResidentProgramsNV = (PFNGLREQUESTRESIDENTPROGRAMSNVPROC)glewGetProcAddress((const GLubyte*)"glRequestResidentProgramsNV")) == NULL) || r;
  r = ((glTrackMatrixNV = (PFNGLTRACKMATRIXNVPROC)glewGetProcAddress((const GLubyte*)"glTrackMatrixNV")) == NULL) || r;
  r = ((glVertexAttrib1dNV = (PFNGLVERTEXATTRIB1DNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1dNV")) == NULL) || r;
  r = ((glVertexAttrib1dvNV = (PFNGLVERTEXATTRIB1DVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1dvNV")) == NULL) || r;
  r = ((glVertexAttrib1fNV = (PFNGLVERTEXATTRIB1FNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1fNV")) == NULL) || r;
  r = ((glVertexAttrib1fvNV = (PFNGLVERTEXATTRIB1FVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1fvNV")) == NULL) || r;
  r = ((glVertexAttrib1sNV = (PFNGLVERTEXATTRIB1SNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1sNV")) == NULL) || r;
  r = ((glVertexAttrib1svNV = (PFNGLVERTEXATTRIB1SVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib1svNV")) == NULL) || r;
  r = ((glVertexAttrib2dNV = (PFNGLVERTEXATTRIB2DNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2dNV")) == NULL) || r;
  r = ((glVertexAttrib2dvNV = (PFNGLVERTEXATTRIB2DVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2dvNV")) == NULL) || r;
  r = ((glVertexAttrib2fNV = (PFNGLVERTEXATTRIB2FNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2fNV")) == NULL) || r;
  r = ((glVertexAttrib2fvNV = (PFNGLVERTEXATTRIB2FVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2fvNV")) == NULL) || r;
  r = ((glVertexAttrib2sNV = (PFNGLVERTEXATTRIB2SNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2sNV")) == NULL) || r;
  r = ((glVertexAttrib2svNV = (PFNGLVERTEXATTRIB2SVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib2svNV")) == NULL) || r;
  r = ((glVertexAttrib3dNV = (PFNGLVERTEXATTRIB3DNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3dNV")) == NULL) || r;
  r = ((glVertexAttrib3dvNV = (PFNGLVERTEXATTRIB3DVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3dvNV")) == NULL) || r;
  r = ((glVertexAttrib3fNV = (PFNGLVERTEXATTRIB3FNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3fNV")) == NULL) || r;
  r = ((glVertexAttrib3fvNV = (PFNGLVERTEXATTRIB3FVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3fvNV")) == NULL) || r;
  r = ((glVertexAttrib3sNV = (PFNGLVERTEXATTRIB3SNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3sNV")) == NULL) || r;
  r = ((glVertexAttrib3svNV = (PFNGLVERTEXATTRIB3SVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib3svNV")) == NULL) || r;
  r = ((glVertexAttrib4dNV = (PFNGLVERTEXATTRIB4DNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4dNV")) == NULL) || r;
  r = ((glVertexAttrib4dvNV = (PFNGLVERTEXATTRIB4DVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4dvNV")) == NULL) || r;
  r = ((glVertexAttrib4fNV = (PFNGLVERTEXATTRIB4FNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4fNV")) == NULL) || r;
  r = ((glVertexAttrib4fvNV = (PFNGLVERTEXATTRIB4FVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4fvNV")) == NULL) || r;
  r = ((glVertexAttrib4sNV = (PFNGLVERTEXATTRIB4SNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4sNV")) == NULL) || r;
  r = ((glVertexAttrib4svNV = (PFNGLVERTEXATTRIB4SVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4svNV")) == NULL) || r;
  r = ((glVertexAttrib4ubNV = (PFNGLVERTEXATTRIB4UBNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4ubNV")) == NULL) || r;
  r = ((glVertexAttrib4ubvNV = (PFNGLVERTEXATTRIB4UBVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttrib4ubvNV")) == NULL) || r;
  r = ((glVertexAttribPointerNV = (PFNGLVERTEXATTRIBPOINTERNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribPointerNV")) == NULL) || r;
  r = ((glVertexAttribs1dvNV = (PFNGLVERTEXATTRIBS1DVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs1dvNV")) == NULL) || r;
  r = ((glVertexAttribs1fvNV = (PFNGLVERTEXATTRIBS1FVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs1fvNV")) == NULL) || r;
  r = ((glVertexAttribs1svNV = (PFNGLVERTEXATTRIBS1SVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs1svNV")) == NULL) || r;
  r = ((glVertexAttribs2dvNV = (PFNGLVERTEXATTRIBS2DVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs2dvNV")) == NULL) || r;
  r = ((glVertexAttribs2fvNV = (PFNGLVERTEXATTRIBS2FVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs2fvNV")) == NULL) || r;
  r = ((glVertexAttribs2svNV = (PFNGLVERTEXATTRIBS2SVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs2svNV")) == NULL) || r;
  r = ((glVertexAttribs3dvNV = (PFNGLVERTEXATTRIBS3DVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs3dvNV")) == NULL) || r;
  r = ((glVertexAttribs3fvNV = (PFNGLVERTEXATTRIBS3FVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs3fvNV")) == NULL) || r;
  r = ((glVertexAttribs3svNV = (PFNGLVERTEXATTRIBS3SVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs3svNV")) == NULL) || r;
  r = ((glVertexAttribs4dvNV = (PFNGLVERTEXATTRIBS4DVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs4dvNV")) == NULL) || r;
  r = ((glVertexAttribs4fvNV = (PFNGLVERTEXATTRIBS4FVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs4fvNV")) == NULL) || r;
  r = ((glVertexAttribs4svNV = (PFNGLVERTEXATTRIBS4SVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs4svNV")) == NULL) || r;
  r = ((glVertexAttribs4ubvNV = (PFNGLVERTEXATTRIBS4UBVNVPROC)glewGetProcAddress((const GLubyte*)"glVertexAttribs4ubvNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_vertex_program */

#ifdef GL_NV_video_capture

static GLboolean _glewInit_GL_NV_video_capture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glBeginVideoCaptureNV = (PFNGLBEGINVIDEOCAPTURENVPROC)glewGetProcAddress((const GLubyte*)"glBeginVideoCaptureNV")) == NULL) || r;
  r = ((glBindVideoCaptureStreamBufferNV = (PFNGLBINDVIDEOCAPTURESTREAMBUFFERNVPROC)glewGetProcAddress((const GLubyte*)"glBindVideoCaptureStreamBufferNV")) == NULL) || r;
  r = ((glBindVideoCaptureStreamTextureNV = (PFNGLBINDVIDEOCAPTURESTREAMTEXTURENVPROC)glewGetProcAddress((const GLubyte*)"glBindVideoCaptureStreamTextureNV")) == NULL) || r;
  r = ((glEndVideoCaptureNV = (PFNGLENDVIDEOCAPTURENVPROC)glewGetProcAddress((const GLubyte*)"glEndVideoCaptureNV")) == NULL) || r;
  r = ((glGetVideoCaptureStreamdvNV = (PFNGLGETVIDEOCAPTURESTREAMDVNVPROC)glewGetProcAddress((const GLubyte*)"glGetVideoCaptureStreamdvNV")) == NULL) || r;
  r = ((glGetVideoCaptureStreamfvNV = (PFNGLGETVIDEOCAPTURESTREAMFVNVPROC)glewGetProcAddress((const GLubyte*)"glGetVideoCaptureStreamfvNV")) == NULL) || r;
  r = ((glGetVideoCaptureStreamivNV = (PFNGLGETVIDEOCAPTURESTREAMIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetVideoCaptureStreamivNV")) == NULL) || r;
  r = ((glGetVideoCaptureivNV = (PFNGLGETVIDEOCAPTUREIVNVPROC)glewGetProcAddress((const GLubyte*)"glGetVideoCaptureivNV")) == NULL) || r;
  r = ((glVideoCaptureNV = (PFNGLVIDEOCAPTURENVPROC)glewGetProcAddress((const GLubyte*)"glVideoCaptureNV")) == NULL) || r;
  r = ((glVideoCaptureStreamParameterdvNV = (PFNGLVIDEOCAPTURESTREAMPARAMETERDVNVPROC)glewGetProcAddress((const GLubyte*)"glVideoCaptureStreamParameterdvNV")) == NULL) || r;
  r = ((glVideoCaptureStreamParameterfvNV = (PFNGLVIDEOCAPTURESTREAMPARAMETERFVNVPROC)glewGetProcAddress((const GLubyte*)"glVideoCaptureStreamParameterfvNV")) == NULL) || r;
  r = ((glVideoCaptureStreamParameterivNV = (PFNGLVIDEOCAPTURESTREAMPARAMETERIVNVPROC)glewGetProcAddress((const GLubyte*)"glVideoCaptureStreamParameterivNV")) == NULL) || r;

  return r;
}

#endif /* GL_NV_video_capture */

#ifdef GL_OES_single_precision

static GLboolean _glewInit_GL_OES_single_precision (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClearDepthfOES = (PFNGLCLEARDEPTHFOESPROC)glewGetProcAddress((const GLubyte*)"glClearDepthfOES")) == NULL) || r;
  r = ((glClipPlanefOES = (PFNGLCLIPPLANEFOESPROC)glewGetProcAddress((const GLubyte*)"glClipPlanefOES")) == NULL) || r;
  r = ((glDepthRangefOES = (PFNGLDEPTHRANGEFOESPROC)glewGetProcAddress((const GLubyte*)"glDepthRangefOES")) == NULL) || r;
  r = ((glFrustumfOES = (PFNGLFRUSTUMFOESPROC)glewGetProcAddress((const GLubyte*)"glFrustumfOES")) == NULL) || r;
  r = ((glGetClipPlanefOES = (PFNGLGETCLIPPLANEFOESPROC)glewGetProcAddress((const GLubyte*)"glGetClipPlanefOES")) == NULL) || r;
  r = ((glOrthofOES = (PFNGLORTHOFOESPROC)glewGetProcAddress((const GLubyte*)"glOrthofOES")) == NULL) || r;

  return r;
}

#endif /* GL_OES_single_precision */

#ifdef GL_OVR_multiview

static GLboolean _glewInit_GL_OVR_multiview (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFramebufferTextureMultiviewOVR = (PFNGLFRAMEBUFFERTEXTUREMULTIVIEWOVRPROC)glewGetProcAddress((const GLubyte*)"glFramebufferTextureMultiviewOVR")) == NULL) || r;

  return r;
}

#endif /* GL_OVR_multiview */

#ifdef GL_REGAL_ES1_0_compatibility

static GLboolean _glewInit_GL_REGAL_ES1_0_compatibility (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glAlphaFuncx = (PFNGLALPHAFUNCXPROC)glewGetProcAddress((const GLubyte*)"glAlphaFuncx")) == NULL) || r;
  r = ((glClearColorx = (PFNGLCLEARCOLORXPROC)glewGetProcAddress((const GLubyte*)"glClearColorx")) == NULL) || r;
  r = ((glClearDepthx = (PFNGLCLEARDEPTHXPROC)glewGetProcAddress((const GLubyte*)"glClearDepthx")) == NULL) || r;
  r = ((glColor4x = (PFNGLCOLOR4XPROC)glewGetProcAddress((const GLubyte*)"glColor4x")) == NULL) || r;
  r = ((glDepthRangex = (PFNGLDEPTHRANGEXPROC)glewGetProcAddress((const GLubyte*)"glDepthRangex")) == NULL) || r;
  r = ((glFogx = (PFNGLFOGXPROC)glewGetProcAddress((const GLubyte*)"glFogx")) == NULL) || r;
  r = ((glFogxv = (PFNGLFOGXVPROC)glewGetProcAddress((const GLubyte*)"glFogxv")) == NULL) || r;
  r = ((glFrustumf = (PFNGLFRUSTUMFPROC)glewGetProcAddress((const GLubyte*)"glFrustumf")) == NULL) || r;
  r = ((glFrustumx = (PFNGLFRUSTUMXPROC)glewGetProcAddress((const GLubyte*)"glFrustumx")) == NULL) || r;
  r = ((glLightModelx = (PFNGLLIGHTMODELXPROC)glewGetProcAddress((const GLubyte*)"glLightModelx")) == NULL) || r;
  r = ((glLightModelxv = (PFNGLLIGHTMODELXVPROC)glewGetProcAddress((const GLubyte*)"glLightModelxv")) == NULL) || r;
  r = ((glLightx = (PFNGLLIGHTXPROC)glewGetProcAddress((const GLubyte*)"glLightx")) == NULL) || r;
  r = ((glLightxv = (PFNGLLIGHTXVPROC)glewGetProcAddress((const GLubyte*)"glLightxv")) == NULL) || r;
  r = ((glLineWidthx = (PFNGLLINEWIDTHXPROC)glewGetProcAddress((const GLubyte*)"glLineWidthx")) == NULL) || r;
  r = ((glLoadMatrixx = (PFNGLLOADMATRIXXPROC)glewGetProcAddress((const GLubyte*)"glLoadMatrixx")) == NULL) || r;
  r = ((glMaterialx = (PFNGLMATERIALXPROC)glewGetProcAddress((const GLubyte*)"glMaterialx")) == NULL) || r;
  r = ((glMaterialxv = (PFNGLMATERIALXVPROC)glewGetProcAddress((const GLubyte*)"glMaterialxv")) == NULL) || r;
  r = ((glMultMatrixx = (PFNGLMULTMATRIXXPROC)glewGetProcAddress((const GLubyte*)"glMultMatrixx")) == NULL) || r;
  r = ((glMultiTexCoord4x = (PFNGLMULTITEXCOORD4XPROC)glewGetProcAddress((const GLubyte*)"glMultiTexCoord4x")) == NULL) || r;
  r = ((glNormal3x = (PFNGLNORMAL3XPROC)glewGetProcAddress((const GLubyte*)"glNormal3x")) == NULL) || r;
  r = ((glOrthof = (PFNGLORTHOFPROC)glewGetProcAddress((const GLubyte*)"glOrthof")) == NULL) || r;
  r = ((glOrthox = (PFNGLORTHOXPROC)glewGetProcAddress((const GLubyte*)"glOrthox")) == NULL) || r;
  r = ((glPointSizex = (PFNGLPOINTSIZEXPROC)glewGetProcAddress((const GLubyte*)"glPointSizex")) == NULL) || r;
  r = ((glPolygonOffsetx = (PFNGLPOLYGONOFFSETXPROC)glewGetProcAddress((const GLubyte*)"glPolygonOffsetx")) == NULL) || r;
  r = ((glRotatex = (PFNGLROTATEXPROC)glewGetProcAddress((const GLubyte*)"glRotatex")) == NULL) || r;
  r = ((glSampleCoveragex = (PFNGLSAMPLECOVERAGEXPROC)glewGetProcAddress((const GLubyte*)"glSampleCoveragex")) == NULL) || r;
  r = ((glScalex = (PFNGLSCALEXPROC)glewGetProcAddress((const GLubyte*)"glScalex")) == NULL) || r;
  r = ((glTexEnvx = (PFNGLTEXENVXPROC)glewGetProcAddress((const GLubyte*)"glTexEnvx")) == NULL) || r;
  r = ((glTexEnvxv = (PFNGLTEXENVXVPROC)glewGetProcAddress((const GLubyte*)"glTexEnvxv")) == NULL) || r;
  r = ((glTexParameterx = (PFNGLTEXPARAMETERXPROC)glewGetProcAddress((const GLubyte*)"glTexParameterx")) == NULL) || r;
  r = ((glTranslatex = (PFNGLTRANSLATEXPROC)glewGetProcAddress((const GLubyte*)"glTranslatex")) == NULL) || r;

  return r;
}

#endif /* GL_REGAL_ES1_0_compatibility */

#ifdef GL_REGAL_ES1_1_compatibility

static GLboolean _glewInit_GL_REGAL_ES1_1_compatibility (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glClipPlanef = (PFNGLCLIPPLANEFPROC)glewGetProcAddress((const GLubyte*)"glClipPlanef")) == NULL) || r;
  r = ((glClipPlanex = (PFNGLCLIPPLANEXPROC)glewGetProcAddress((const GLubyte*)"glClipPlanex")) == NULL) || r;
  r = ((glGetClipPlanef = (PFNGLGETCLIPPLANEFPROC)glewGetProcAddress((const GLubyte*)"glGetClipPlanef")) == NULL) || r;
  r = ((glGetClipPlanex = (PFNGLGETCLIPPLANEXPROC)glewGetProcAddress((const GLubyte*)"glGetClipPlanex")) == NULL) || r;
  r = ((glGetFixedv = (PFNGLGETFIXEDVPROC)glewGetProcAddress((const GLubyte*)"glGetFixedv")) == NULL) || r;
  r = ((glGetLightxv = (PFNGLGETLIGHTXVPROC)glewGetProcAddress((const GLubyte*)"glGetLightxv")) == NULL) || r;
  r = ((glGetMaterialxv = (PFNGLGETMATERIALXVPROC)glewGetProcAddress((const GLubyte*)"glGetMaterialxv")) == NULL) || r;
  r = ((glGetTexEnvxv = (PFNGLGETTEXENVXVPROC)glewGetProcAddress((const GLubyte*)"glGetTexEnvxv")) == NULL) || r;
  r = ((glGetTexParameterxv = (PFNGLGETTEXPARAMETERXVPROC)glewGetProcAddress((const GLubyte*)"glGetTexParameterxv")) == NULL) || r;
  r = ((glPointParameterx = (PFNGLPOINTPARAMETERXPROC)glewGetProcAddress((const GLubyte*)"glPointParameterx")) == NULL) || r;
  r = ((glPointParameterxv = (PFNGLPOINTPARAMETERXVPROC)glewGetProcAddress((const GLubyte*)"glPointParameterxv")) == NULL) || r;
  r = ((glPointSizePointerOES = (PFNGLPOINTSIZEPOINTEROESPROC)glewGetProcAddress((const GLubyte*)"glPointSizePointerOES")) == NULL) || r;
  r = ((glTexParameterxv = (PFNGLTEXPARAMETERXVPROC)glewGetProcAddress((const GLubyte*)"glTexParameterxv")) == NULL) || r;

  return r;
}

#endif /* GL_REGAL_ES1_1_compatibility */

#ifdef GL_REGAL_error_string

static GLboolean _glewInit_GL_REGAL_error_string (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glErrorStringREGAL = (PFNGLERRORSTRINGREGALPROC)glewGetProcAddress((const GLubyte*)"glErrorStringREGAL")) == NULL) || r;

  return r;
}

#endif /* GL_REGAL_error_string */

#ifdef GL_REGAL_extension_query

static GLboolean _glewInit_GL_REGAL_extension_query (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetExtensionREGAL = (PFNGLGETEXTENSIONREGALPROC)glewGetProcAddress((const GLubyte*)"glGetExtensionREGAL")) == NULL) || r;
  r = ((glIsSupportedREGAL = (PFNGLISSUPPORTEDREGALPROC)glewGetProcAddress((const GLubyte*)"glIsSupportedREGAL")) == NULL) || r;

  return r;
}

#endif /* GL_REGAL_extension_query */

#ifdef GL_REGAL_log

static GLboolean _glewInit_GL_REGAL_log (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glLogMessageCallbackREGAL = (PFNGLLOGMESSAGECALLBACKREGALPROC)glewGetProcAddress((const GLubyte*)"glLogMessageCallbackREGAL")) == NULL) || r;

  return r;
}

#endif /* GL_REGAL_log */

#ifdef GL_REGAL_proc_address

static GLboolean _glewInit_GL_REGAL_proc_address (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetProcAddressREGAL = (PFNGLGETPROCADDRESSREGALPROC)glewGetProcAddress((const GLubyte*)"glGetProcAddressREGAL")) == NULL) || r;

  return r;
}

#endif /* GL_REGAL_proc_address */

#ifdef GL_SGIS_detail_texture

static GLboolean _glewInit_GL_SGIS_detail_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glDetailTexFuncSGIS = (PFNGLDETAILTEXFUNCSGISPROC)glewGetProcAddress((const GLubyte*)"glDetailTexFuncSGIS")) == NULL) || r;
  r = ((glGetDetailTexFuncSGIS = (PFNGLGETDETAILTEXFUNCSGISPROC)glewGetProcAddress((const GLubyte*)"glGetDetailTexFuncSGIS")) == NULL) || r;

  return r;
}

#endif /* GL_SGIS_detail_texture */

#ifdef GL_SGIS_fog_function

static GLboolean _glewInit_GL_SGIS_fog_function (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFogFuncSGIS = (PFNGLFOGFUNCSGISPROC)glewGetProcAddress((const GLubyte*)"glFogFuncSGIS")) == NULL) || r;
  r = ((glGetFogFuncSGIS = (PFNGLGETFOGFUNCSGISPROC)glewGetProcAddress((const GLubyte*)"glGetFogFuncSGIS")) == NULL) || r;

  return r;
}

#endif /* GL_SGIS_fog_function */

#ifdef GL_SGIS_multisample

static GLboolean _glewInit_GL_SGIS_multisample (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glSampleMaskSGIS = (PFNGLSAMPLEMASKSGISPROC)glewGetProcAddress((const GLubyte*)"glSampleMaskSGIS")) == NULL) || r;
  r = ((glSamplePatternSGIS = (PFNGLSAMPLEPATTERNSGISPROC)glewGetProcAddress((const GLubyte*)"glSamplePatternSGIS")) == NULL) || r;

  return r;
}

#endif /* GL_SGIS_multisample */

#ifdef GL_SGIS_sharpen_texture

static GLboolean _glewInit_GL_SGIS_sharpen_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetSharpenTexFuncSGIS = (PFNGLGETSHARPENTEXFUNCSGISPROC)glewGetProcAddress((const GLubyte*)"glGetSharpenTexFuncSGIS")) == NULL) || r;
  r = ((glSharpenTexFuncSGIS = (PFNGLSHARPENTEXFUNCSGISPROC)glewGetProcAddress((const GLubyte*)"glSharpenTexFuncSGIS")) == NULL) || r;

  return r;
}

#endif /* GL_SGIS_sharpen_texture */

#ifdef GL_SGIS_texture4D

static GLboolean _glewInit_GL_SGIS_texture4D (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTexImage4DSGIS = (PFNGLTEXIMAGE4DSGISPROC)glewGetProcAddress((const GLubyte*)"glTexImage4DSGIS")) == NULL) || r;
  r = ((glTexSubImage4DSGIS = (PFNGLTEXSUBIMAGE4DSGISPROC)glewGetProcAddress((const GLubyte*)"glTexSubImage4DSGIS")) == NULL) || r;

  return r;
}

#endif /* GL_SGIS_texture4D */

#ifdef GL_SGIS_texture_filter4

static GLboolean _glewInit_GL_SGIS_texture_filter4 (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGetTexFilterFuncSGIS = (PFNGLGETTEXFILTERFUNCSGISPROC)glewGetProcAddress((const GLubyte*)"glGetTexFilterFuncSGIS")) == NULL) || r;
  r = ((glTexFilterFuncSGIS = (PFNGLTEXFILTERFUNCSGISPROC)glewGetProcAddress((const GLubyte*)"glTexFilterFuncSGIS")) == NULL) || r;

  return r;
}

#endif /* GL_SGIS_texture_filter4 */

#ifdef GL_SGIX_async

static GLboolean _glewInit_GL_SGIX_async (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glAsyncMarkerSGIX = (PFNGLASYNCMARKERSGIXPROC)glewGetProcAddress((const GLubyte*)"glAsyncMarkerSGIX")) == NULL) || r;
  r = ((glDeleteAsyncMarkersSGIX = (PFNGLDELETEASYNCMARKERSSGIXPROC)glewGetProcAddress((const GLubyte*)"glDeleteAsyncMarkersSGIX")) == NULL) || r;
  r = ((glFinishAsyncSGIX = (PFNGLFINISHASYNCSGIXPROC)glewGetProcAddress((const GLubyte*)"glFinishAsyncSGIX")) == NULL) || r;
  r = ((glGenAsyncMarkersSGIX = (PFNGLGENASYNCMARKERSSGIXPROC)glewGetProcAddress((const GLubyte*)"glGenAsyncMarkersSGIX")) == NULL) || r;
  r = ((glIsAsyncMarkerSGIX = (PFNGLISASYNCMARKERSGIXPROC)glewGetProcAddress((const GLubyte*)"glIsAsyncMarkerSGIX")) == NULL) || r;
  r = ((glPollAsyncSGIX = (PFNGLPOLLASYNCSGIXPROC)glewGetProcAddress((const GLubyte*)"glPollAsyncSGIX")) == NULL) || r;

  return r;
}

#endif /* GL_SGIX_async */

#ifdef GL_SGIX_flush_raster

static GLboolean _glewInit_GL_SGIX_flush_raster (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFlushRasterSGIX = (PFNGLFLUSHRASTERSGIXPROC)glewGetProcAddress((const GLubyte*)"glFlushRasterSGIX")) == NULL) || r;

  return r;
}

#endif /* GL_SGIX_flush_raster */

#ifdef GL_SGIX_fog_texture

static GLboolean _glewInit_GL_SGIX_fog_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTextureFogSGIX = (PFNGLTEXTUREFOGSGIXPROC)glewGetProcAddress((const GLubyte*)"glTextureFogSGIX")) == NULL) || r;

  return r;
}

#endif /* GL_SGIX_fog_texture */

#ifdef GL_SGIX_fragment_specular_lighting

static GLboolean _glewInit_GL_SGIX_fragment_specular_lighting (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFragmentColorMaterialSGIX = (PFNGLFRAGMENTCOLORMATERIALSGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentColorMaterialSGIX")) == NULL) || r;
  r = ((glFragmentLightModelfSGIX = (PFNGLFRAGMENTLIGHTMODELFSGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightModelfSGIX")) == NULL) || r;
  r = ((glFragmentLightModelfvSGIX = (PFNGLFRAGMENTLIGHTMODELFVSGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightModelfvSGIX")) == NULL) || r;
  r = ((glFragmentLightModeliSGIX = (PFNGLFRAGMENTLIGHTMODELISGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightModeliSGIX")) == NULL) || r;
  r = ((glFragmentLightModelivSGIX = (PFNGLFRAGMENTLIGHTMODELIVSGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightModelivSGIX")) == NULL) || r;
  r = ((glFragmentLightfSGIX = (PFNGLFRAGMENTLIGHTFSGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightfSGIX")) == NULL) || r;
  r = ((glFragmentLightfvSGIX = (PFNGLFRAGMENTLIGHTFVSGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightfvSGIX")) == NULL) || r;
  r = ((glFragmentLightiSGIX = (PFNGLFRAGMENTLIGHTISGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightiSGIX")) == NULL) || r;
  r = ((glFragmentLightivSGIX = (PFNGLFRAGMENTLIGHTIVSGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentLightivSGIX")) == NULL) || r;
  r = ((glFragmentMaterialfSGIX = (PFNGLFRAGMENTMATERIALFSGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentMaterialfSGIX")) == NULL) || r;
  r = ((glFragmentMaterialfvSGIX = (PFNGLFRAGMENTMATERIALFVSGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentMaterialfvSGIX")) == NULL) || r;
  r = ((glFragmentMaterialiSGIX = (PFNGLFRAGMENTMATERIALISGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentMaterialiSGIX")) == NULL) || r;
  r = ((glFragmentMaterialivSGIX = (PFNGLFRAGMENTMATERIALIVSGIXPROC)glewGetProcAddress((const GLubyte*)"glFragmentMaterialivSGIX")) == NULL) || r;
  r = ((glGetFragmentLightfvSGIX = (PFNGLGETFRAGMENTLIGHTFVSGIXPROC)glewGetProcAddress((const GLubyte*)"glGetFragmentLightfvSGIX")) == NULL) || r;
  r = ((glGetFragmentLightivSGIX = (PFNGLGETFRAGMENTLIGHTIVSGIXPROC)glewGetProcAddress((const GLubyte*)"glGetFragmentLightivSGIX")) == NULL) || r;
  r = ((glGetFragmentMaterialfvSGIX = (PFNGLGETFRAGMENTMATERIALFVSGIXPROC)glewGetProcAddress((const GLubyte*)"glGetFragmentMaterialfvSGIX")) == NULL) || r;
  r = ((glGetFragmentMaterialivSGIX = (PFNGLGETFRAGMENTMATERIALIVSGIXPROC)glewGetProcAddress((const GLubyte*)"glGetFragmentMaterialivSGIX")) == NULL) || r;

  return r;
}

#endif /* GL_SGIX_fragment_specular_lighting */

#ifdef GL_SGIX_framezoom

static GLboolean _glewInit_GL_SGIX_framezoom (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFrameZoomSGIX = (PFNGLFRAMEZOOMSGIXPROC)glewGetProcAddress((const GLubyte*)"glFrameZoomSGIX")) == NULL) || r;

  return r;
}

#endif /* GL_SGIX_framezoom */

#ifdef GL_SGIX_pixel_texture

static GLboolean _glewInit_GL_SGIX_pixel_texture (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glPixelTexGenSGIX = (PFNGLPIXELTEXGENSGIXPROC)glewGetProcAddress((const GLubyte*)"glPixelTexGenSGIX")) == NULL) || r;

  return r;
}

#endif /* GL_SGIX_pixel_texture */

#ifdef GL_SGIX_reference_plane

static GLboolean _glewInit_GL_SGIX_reference_plane (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glReferencePlaneSGIX = (PFNGLREFERENCEPLANESGIXPROC)glewGetProcAddress((const GLubyte*)"glReferencePlaneSGIX")) == NULL) || r;

  return r;
}

#endif /* GL_SGIX_reference_plane */

#ifdef GL_SGIX_sprite

static GLboolean _glewInit_GL_SGIX_sprite (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glSpriteParameterfSGIX = (PFNGLSPRITEPARAMETERFSGIXPROC)glewGetProcAddress((const GLubyte*)"glSpriteParameterfSGIX")) == NULL) || r;
  r = ((glSpriteParameterfvSGIX = (PFNGLSPRITEPARAMETERFVSGIXPROC)glewGetProcAddress((const GLubyte*)"glSpriteParameterfvSGIX")) == NULL) || r;
  r = ((glSpriteParameteriSGIX = (PFNGLSPRITEPARAMETERISGIXPROC)glewGetProcAddress((const GLubyte*)"glSpriteParameteriSGIX")) == NULL) || r;
  r = ((glSpriteParameterivSGIX = (PFNGLSPRITEPARAMETERIVSGIXPROC)glewGetProcAddress((const GLubyte*)"glSpriteParameterivSGIX")) == NULL) || r;

  return r;
}

#endif /* GL_SGIX_sprite */

#ifdef GL_SGIX_tag_sample_buffer

static GLboolean _glewInit_GL_SGIX_tag_sample_buffer (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glTagSampleBufferSGIX = (PFNGLTAGSAMPLEBUFFERSGIXPROC)glewGetProcAddress((const GLubyte*)"glTagSampleBufferSGIX")) == NULL) || r;

  return r;
}

#endif /* GL_SGIX_tag_sample_buffer */

#ifdef GL_SGI_color_table

static GLboolean _glewInit_GL_SGI_color_table (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glColorTableParameterfvSGI = (PFNGLCOLORTABLEPARAMETERFVSGIPROC)glewGetProcAddress((const GLubyte*)"glColorTableParameterfvSGI")) == NULL) || r;
  r = ((glColorTableParameterivSGI = (PFNGLCOLORTABLEPARAMETERIVSGIPROC)glewGetProcAddress((const GLubyte*)"glColorTableParameterivSGI")) == NULL) || r;
  r = ((glColorTableSGI = (PFNGLCOLORTABLESGIPROC)glewGetProcAddress((const GLubyte*)"glColorTableSGI")) == NULL) || r;
  r = ((glCopyColorTableSGI = (PFNGLCOPYCOLORTABLESGIPROC)glewGetProcAddress((const GLubyte*)"glCopyColorTableSGI")) == NULL) || r;
  r = ((glGetColorTableParameterfvSGI = (PFNGLGETCOLORTABLEPARAMETERFVSGIPROC)glewGetProcAddress((const GLubyte*)"glGetColorTableParameterfvSGI")) == NULL) || r;
  r = ((glGetColorTableParameterivSGI = (PFNGLGETCOLORTABLEPARAMETERIVSGIPROC)glewGetProcAddress((const GLubyte*)"glGetColorTableParameterivSGI")) == NULL) || r;
  r = ((glGetColorTableSGI = (PFNGLGETCOLORTABLESGIPROC)glewGetProcAddress((const GLubyte*)"glGetColorTableSGI")) == NULL) || r;

  return r;
}

#endif /* GL_SGI_color_table */

#ifdef GL_SUNX_constant_data

static GLboolean _glewInit_GL_SUNX_constant_data (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glFinishTextureSUNX = (PFNGLFINISHTEXTURESUNXPROC)glewGetProcAddress((const GLubyte*)"glFinishTextureSUNX")) == NULL) || r;

  return r;
}

#endif /* GL_SUNX_constant_data */

#ifdef GL_SUN_global_alpha

static GLboolean _glewInit_GL_SUN_global_alpha (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glGlobalAlphaFactorbSUN = (PFNGLGLOBALALPHAFACTORBSUNPROC)glewGetProcAddress((const GLubyte*)"glGlobalAlphaFactorbSUN")) == NULL) || r;
  r = ((glGlobalAlphaFactordSUN = (PFNGLGLOBALALPHAFACTORDSUNPROC)glewGetProcAddress((const GLubyte*)"glGlobalAlphaFactordSUN")) == NULL) || r;
  r = ((glGlobalAlphaFactorfSUN = (PFNGLGLOBALALPHAFACTORFSUNPROC)glewGetProcAddress((const GLubyte*)"glGlobalAlphaFactorfSUN")) == NULL) || r;
  r = ((glGlobalAlphaFactoriSUN = (PFNGLGLOBALALPHAFACTORISUNPROC)glewGetProcAddress((const GLubyte*)"glGlobalAlphaFactoriSUN")) == NULL) || r;
  r = ((glGlobalAlphaFactorsSUN = (PFNGLGLOBALALPHAFACTORSSUNPROC)glewGetProcAddress((const GLubyte*)"glGlobalAlphaFactorsSUN")) == NULL) || r;
  r = ((glGlobalAlphaFactorubSUN = (PFNGLGLOBALALPHAFACTORUBSUNPROC)glewGetProcAddress((const GLubyte*)"glGlobalAlphaFactorubSUN")) == NULL) || r;
  r = ((glGlobalAlphaFactoruiSUN = (PFNGLGLOBALALPHAFACTORUISUNPROC)glewGetProcAddress((const GLubyte*)"glGlobalAlphaFactoruiSUN")) == NULL) || r;
  r = ((glGlobalAlphaFactorusSUN = (PFNGLGLOBALALPHAFACTORUSSUNPROC)glewGetProcAddress((const GLubyte*)"glGlobalAlphaFactorusSUN")) == NULL) || r;

  return r;
}

#endif /* GL_SUN_global_alpha */

#ifdef GL_SUN_read_video_pixels

static GLboolean _glewInit_GL_SUN_read_video_pixels (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glReadVideoPixelsSUN = (PFNGLREADVIDEOPIXELSSUNPROC)glewGetProcAddress((const GLubyte*)"glReadVideoPixelsSUN")) == NULL) || r;

  return r;
}

#endif /* GL_SUN_read_video_pixels */

#ifdef GL_SUN_triangle_list

static GLboolean _glewInit_GL_SUN_triangle_list (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glReplacementCodePointerSUN = (PFNGLREPLACEMENTCODEPOINTERSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodePointerSUN")) == NULL) || r;
  r = ((glReplacementCodeubSUN = (PFNGLREPLACEMENTCODEUBSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeubSUN")) == NULL) || r;
  r = ((glReplacementCodeubvSUN = (PFNGLREPLACEMENTCODEUBVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeubvSUN")) == NULL) || r;
  r = ((glReplacementCodeuiSUN = (PFNGLREPLACEMENTCODEUISUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiSUN")) == NULL) || r;
  r = ((glReplacementCodeuivSUN = (PFNGLREPLACEMENTCODEUIVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuivSUN")) == NULL) || r;
  r = ((glReplacementCodeusSUN = (PFNGLREPLACEMENTCODEUSSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeusSUN")) == NULL) || r;
  r = ((glReplacementCodeusvSUN = (PFNGLREPLACEMENTCODEUSVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeusvSUN")) == NULL) || r;

  return r;
}

#endif /* GL_SUN_triangle_list */

#ifdef GL_SUN_vertex

static GLboolean _glewInit_GL_SUN_vertex (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glColor3fVertex3fSUN = (PFNGLCOLOR3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glColor3fVertex3fSUN")) == NULL) || r;
  r = ((glColor3fVertex3fvSUN = (PFNGLCOLOR3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glColor3fVertex3fvSUN")) == NULL) || r;
  r = ((glColor4fNormal3fVertex3fSUN = (PFNGLCOLOR4FNORMAL3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glColor4fNormal3fVertex3fSUN")) == NULL) || r;
  r = ((glColor4fNormal3fVertex3fvSUN = (PFNGLCOLOR4FNORMAL3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glColor4fNormal3fVertex3fvSUN")) == NULL) || r;
  r = ((glColor4ubVertex2fSUN = (PFNGLCOLOR4UBVERTEX2FSUNPROC)glewGetProcAddress((const GLubyte*)"glColor4ubVertex2fSUN")) == NULL) || r;
  r = ((glColor4ubVertex2fvSUN = (PFNGLCOLOR4UBVERTEX2FVSUNPROC)glewGetProcAddress((const GLubyte*)"glColor4ubVertex2fvSUN")) == NULL) || r;
  r = ((glColor4ubVertex3fSUN = (PFNGLCOLOR4UBVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glColor4ubVertex3fSUN")) == NULL) || r;
  r = ((glColor4ubVertex3fvSUN = (PFNGLCOLOR4UBVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glColor4ubVertex3fvSUN")) == NULL) || r;
  r = ((glNormal3fVertex3fSUN = (PFNGLNORMAL3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glNormal3fVertex3fSUN")) == NULL) || r;
  r = ((glNormal3fVertex3fvSUN = (PFNGLNORMAL3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glNormal3fVertex3fvSUN")) == NULL) || r;
  r = ((glReplacementCodeuiColor3fVertex3fSUN = (PFNGLREPLACEMENTCODEUICOLOR3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiColor3fVertex3fSUN")) == NULL) || r;
  r = ((glReplacementCodeuiColor3fVertex3fvSUN = (PFNGLREPLACEMENTCODEUICOLOR3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiColor3fVertex3fvSUN")) == NULL) || r;
  r = ((glReplacementCodeuiColor4fNormal3fVertex3fSUN = (PFNGLREPLACEMENTCODEUICOLOR4FNORMAL3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiColor4fNormal3fVertex3fSUN")) == NULL) || r;
  r = ((glReplacementCodeuiColor4fNormal3fVertex3fvSUN = (PFNGLREPLACEMENTCODEUICOLOR4FNORMAL3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiColor4fNormal3fVertex3fvSUN")) == NULL) || r;
  r = ((glReplacementCodeuiColor4ubVertex3fSUN = (PFNGLREPLACEMENTCODEUICOLOR4UBVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiColor4ubVertex3fSUN")) == NULL) || r;
  r = ((glReplacementCodeuiColor4ubVertex3fvSUN = (PFNGLREPLACEMENTCODEUICOLOR4UBVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiColor4ubVertex3fvSUN")) == NULL) || r;
  r = ((glReplacementCodeuiNormal3fVertex3fSUN = (PFNGLREPLACEMENTCODEUINORMAL3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiNormal3fVertex3fSUN")) == NULL) || r;
  r = ((glReplacementCodeuiNormal3fVertex3fvSUN = (PFNGLREPLACEMENTCODEUINORMAL3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiNormal3fVertex3fvSUN")) == NULL) || r;
  r = ((glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FCOLOR4FNORMAL3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fSUN")) == NULL) || r;
  r = ((glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fvSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FCOLOR4FNORMAL3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fvSUN")) == NULL) || r;
  r = ((glReplacementCodeuiTexCoord2fNormal3fVertex3fSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FNORMAL3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiTexCoord2fNormal3fVertex3fSUN")) == NULL) || r;
  r = ((glReplacementCodeuiTexCoord2fNormal3fVertex3fvSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FNORMAL3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiTexCoord2fNormal3fVertex3fvSUN")) == NULL) || r;
  r = ((glReplacementCodeuiTexCoord2fVertex3fSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiTexCoord2fVertex3fSUN")) == NULL) || r;
  r = ((glReplacementCodeuiTexCoord2fVertex3fvSUN = (PFNGLREPLACEMENTCODEUITEXCOORD2FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiTexCoord2fVertex3fvSUN")) == NULL) || r;
  r = ((glReplacementCodeuiVertex3fSUN = (PFNGLREPLACEMENTCODEUIVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiVertex3fSUN")) == NULL) || r;
  r = ((glReplacementCodeuiVertex3fvSUN = (PFNGLREPLACEMENTCODEUIVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glReplacementCodeuiVertex3fvSUN")) == NULL) || r;
  r = ((glTexCoord2fColor3fVertex3fSUN = (PFNGLTEXCOORD2FCOLOR3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2fColor3fVertex3fSUN")) == NULL) || r;
  r = ((glTexCoord2fColor3fVertex3fvSUN = (PFNGLTEXCOORD2FCOLOR3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2fColor3fVertex3fvSUN")) == NULL) || r;
  r = ((glTexCoord2fColor4fNormal3fVertex3fSUN = (PFNGLTEXCOORD2FCOLOR4FNORMAL3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2fColor4fNormal3fVertex3fSUN")) == NULL) || r;
  r = ((glTexCoord2fColor4fNormal3fVertex3fvSUN = (PFNGLTEXCOORD2FCOLOR4FNORMAL3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2fColor4fNormal3fVertex3fvSUN")) == NULL) || r;
  r = ((glTexCoord2fColor4ubVertex3fSUN = (PFNGLTEXCOORD2FCOLOR4UBVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2fColor4ubVertex3fSUN")) == NULL) || r;
  r = ((glTexCoord2fColor4ubVertex3fvSUN = (PFNGLTEXCOORD2FCOLOR4UBVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2fColor4ubVertex3fvSUN")) == NULL) || r;
  r = ((glTexCoord2fNormal3fVertex3fSUN = (PFNGLTEXCOORD2FNORMAL3FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2fNormal3fVertex3fSUN")) == NULL) || r;
  r = ((glTexCoord2fNormal3fVertex3fvSUN = (PFNGLTEXCOORD2FNORMAL3FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2fNormal3fVertex3fvSUN")) == NULL) || r;
  r = ((glTexCoord2fVertex3fSUN = (PFNGLTEXCOORD2FVERTEX3FSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2fVertex3fSUN")) == NULL) || r;
  r = ((glTexCoord2fVertex3fvSUN = (PFNGLTEXCOORD2FVERTEX3FVSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord2fVertex3fvSUN")) == NULL) || r;
  r = ((glTexCoord4fColor4fNormal3fVertex4fSUN = (PFNGLTEXCOORD4FCOLOR4FNORMAL3FVERTEX4FSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord4fColor4fNormal3fVertex4fSUN")) == NULL) || r;
  r = ((glTexCoord4fColor4fNormal3fVertex4fvSUN = (PFNGLTEXCOORD4FCOLOR4FNORMAL3FVERTEX4FVSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord4fColor4fNormal3fVertex4fvSUN")) == NULL) || r;
  r = ((glTexCoord4fVertex4fSUN = (PFNGLTEXCOORD4FVERTEX4FSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord4fVertex4fSUN")) == NULL) || r;
  r = ((glTexCoord4fVertex4fvSUN = (PFNGLTEXCOORD4FVERTEX4FVSUNPROC)glewGetProcAddress((const GLubyte*)"glTexCoord4fVertex4fvSUN")) == NULL) || r;

  return r;
}

#endif /* GL_SUN_vertex */

#ifdef GL_WIN_swap_hint

static GLboolean _glewInit_GL_WIN_swap_hint (GLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glAddSwapHintRectWIN = (PFNGLADDSWAPHINTRECTWINPROC)glewGetProcAddress((const GLubyte*)"glAddSwapHintRectWIN")) == NULL) || r;

  return r;
}

#endif /* GL_WIN_swap_hint */

/* ------------------------------------------------------------------------- */

GLboolean GLEWAPIENTRY glewGetExtension (const char* name)
{    
  const GLubyte* start;
  const GLubyte* end;
  start = (const GLubyte*)glGetString(GL_EXTENSIONS);
  if (start == 0)
    return GL_FALSE;
  end = start + _glewStrLen(start);
  return _glewSearchExtension(name, start, end);
}

/* ------------------------------------------------------------------------- */

#ifndef GLEW_MX
static
#endif
GLenum GLEWAPIENTRY glewContextInit (GLEW_CONTEXT_ARG_DEF_LIST)
{
  const GLubyte* s;
  GLuint dot;
  GLint major, minor;
  const GLubyte* extStart;
  const GLubyte* extEnd;
  /* query opengl version */
  s = glGetString(GL_VERSION);
  dot = _glewStrCLen(s, '.');
  if (dot == 0)
    return GLEW_ERROR_NO_GL_VERSION;
  
  major = s[dot-1]-'0';
  minor = s[dot+1]-'0';

  if (minor < 0 || minor > 9)
    minor = 0;
  if (major<0 || major>9)
    return GLEW_ERROR_NO_GL_VERSION;
  

  if (major == 1 && minor == 0)
  {
    return GLEW_ERROR_GL_VERSION_10_ONLY;
  }
  else
  {
    GLEW_VERSION_4_5   = ( major > 4 )                 || ( major == 4 && minor >= 5 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_4_4   = GLEW_VERSION_4_5   == GL_TRUE || ( major == 4 && minor >= 4 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_4_3   = GLEW_VERSION_4_4   == GL_TRUE || ( major == 4 && minor >= 3 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_4_2   = GLEW_VERSION_4_3   == GL_TRUE || ( major == 4 && minor >= 2 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_4_1   = GLEW_VERSION_4_2   == GL_TRUE || ( major == 4 && minor >= 1 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_4_0   = GLEW_VERSION_4_1   == GL_TRUE || ( major == 4               ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_3_3   = GLEW_VERSION_4_0   == GL_TRUE || ( major == 3 && minor >= 3 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_3_2   = GLEW_VERSION_3_3   == GL_TRUE || ( major == 3 && minor >= 2 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_3_1   = GLEW_VERSION_3_2   == GL_TRUE || ( major == 3 && minor >= 1 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_3_0   = GLEW_VERSION_3_1   == GL_TRUE || ( major == 3               ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_2_1   = GLEW_VERSION_3_0   == GL_TRUE || ( major == 2 && minor >= 1 ) ? GL_TRUE : GL_FALSE;    
    GLEW_VERSION_2_0   = GLEW_VERSION_2_1   == GL_TRUE || ( major == 2               ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_1_5   = GLEW_VERSION_2_0   == GL_TRUE || ( major == 1 && minor >= 5 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_1_4   = GLEW_VERSION_1_5   == GL_TRUE || ( major == 1 && minor >= 4 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_1_3   = GLEW_VERSION_1_4   == GL_TRUE || ( major == 1 && minor >= 3 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_1_2_1 = GLEW_VERSION_1_3   == GL_TRUE                                 ? GL_TRUE : GL_FALSE; 
    GLEW_VERSION_1_2   = GLEW_VERSION_1_2_1 == GL_TRUE || ( major == 1 && minor >= 2 ) ? GL_TRUE : GL_FALSE;
    GLEW_VERSION_1_1   = GLEW_VERSION_1_2   == GL_TRUE || ( major == 1 && minor >= 1 ) ? GL_TRUE : GL_FALSE;
  }

  /* query opengl extensions string */
  extStart = glGetString(GL_EXTENSIONS);
  if (extStart == 0)
    extStart = (const GLubyte*)"";
  extEnd = extStart + _glewStrLen(extStart);

  /* initialize extensions */
#ifdef GL_VERSION_1_2
  if (glewExperimental || GLEW_VERSION_1_2) GLEW_VERSION_1_2 = !_glewInit_GL_VERSION_1_2(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_1_2 */
#ifdef GL_VERSION_1_2_1
#endif /* GL_VERSION_1_2_1 */
#ifdef GL_VERSION_1_3
  if (glewExperimental || GLEW_VERSION_1_3) GLEW_VERSION_1_3 = !_glewInit_GL_VERSION_1_3(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_1_3 */
#ifdef GL_VERSION_1_4
  if (glewExperimental || GLEW_VERSION_1_4) GLEW_VERSION_1_4 = !_glewInit_GL_VERSION_1_4(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_1_4 */
#ifdef GL_VERSION_1_5
  if (glewExperimental || GLEW_VERSION_1_5) GLEW_VERSION_1_5 = !_glewInit_GL_VERSION_1_5(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_1_5 */
#ifdef GL_VERSION_2_0
  if (glewExperimental || GLEW_VERSION_2_0) GLEW_VERSION_2_0 = !_glewInit_GL_VERSION_2_0(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_2_0 */
#ifdef GL_VERSION_2_1
  if (glewExperimental || GLEW_VERSION_2_1) GLEW_VERSION_2_1 = !_glewInit_GL_VERSION_2_1(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_2_1 */
#ifdef GL_VERSION_3_0
  if (glewExperimental || GLEW_VERSION_3_0) GLEW_VERSION_3_0 = !_glewInit_GL_VERSION_3_0(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_3_0 */
#ifdef GL_VERSION_3_1
  if (glewExperimental || GLEW_VERSION_3_1) GLEW_VERSION_3_1 = !_glewInit_GL_VERSION_3_1(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_3_1 */
#ifdef GL_VERSION_3_2
  if (glewExperimental || GLEW_VERSION_3_2) GLEW_VERSION_3_2 = !_glewInit_GL_VERSION_3_2(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_3_2 */
#ifdef GL_VERSION_3_3
  if (glewExperimental || GLEW_VERSION_3_3) GLEW_VERSION_3_3 = !_glewInit_GL_VERSION_3_3(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_3_3 */
#ifdef GL_VERSION_4_0
  if (glewExperimental || GLEW_VERSION_4_0) GLEW_VERSION_4_0 = !_glewInit_GL_VERSION_4_0(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_4_0 */
#ifdef GL_VERSION_4_1
#endif /* GL_VERSION_4_1 */
#ifdef GL_VERSION_4_2
#endif /* GL_VERSION_4_2 */
#ifdef GL_VERSION_4_3
#endif /* GL_VERSION_4_3 */
#ifdef GL_VERSION_4_4
#endif /* GL_VERSION_4_4 */
#ifdef GL_VERSION_4_5
  if (glewExperimental || GLEW_VERSION_4_5) GLEW_VERSION_4_5 = !_glewInit_GL_VERSION_4_5(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_VERSION_4_5 */
#ifdef GL_3DFX_multisample
  GLEW_3DFX_multisample = _glewSearchExtension("GL_3DFX_multisample", extStart, extEnd);
#endif /* GL_3DFX_multisample */
#ifdef GL_3DFX_tbuffer
  GLEW_3DFX_tbuffer = _glewSearchExtension("GL_3DFX_tbuffer", extStart, extEnd);
  if (glewExperimental || GLEW_3DFX_tbuffer) GLEW_3DFX_tbuffer = !_glewInit_GL_3DFX_tbuffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_3DFX_tbuffer */
#ifdef GL_3DFX_texture_compression_FXT1
  GLEW_3DFX_texture_compression_FXT1 = _glewSearchExtension("GL_3DFX_texture_compression_FXT1", extStart, extEnd);
#endif /* GL_3DFX_texture_compression_FXT1 */
#ifdef GL_AMD_blend_minmax_factor
  GLEW_AMD_blend_minmax_factor = _glewSearchExtension("GL_AMD_blend_minmax_factor", extStart, extEnd);
#endif /* GL_AMD_blend_minmax_factor */
#ifdef GL_AMD_conservative_depth
  GLEW_AMD_conservative_depth = _glewSearchExtension("GL_AMD_conservative_depth", extStart, extEnd);
#endif /* GL_AMD_conservative_depth */
#ifdef GL_AMD_debug_output
  GLEW_AMD_debug_output = _glewSearchExtension("GL_AMD_debug_output", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_debug_output) GLEW_AMD_debug_output = !_glewInit_GL_AMD_debug_output(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_debug_output */
#ifdef GL_AMD_depth_clamp_separate
  GLEW_AMD_depth_clamp_separate = _glewSearchExtension("GL_AMD_depth_clamp_separate", extStart, extEnd);
#endif /* GL_AMD_depth_clamp_separate */
#ifdef GL_AMD_draw_buffers_blend
  GLEW_AMD_draw_buffers_blend = _glewSearchExtension("GL_AMD_draw_buffers_blend", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_draw_buffers_blend) GLEW_AMD_draw_buffers_blend = !_glewInit_GL_AMD_draw_buffers_blend(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_draw_buffers_blend */
#ifdef GL_AMD_gcn_shader
  GLEW_AMD_gcn_shader = _glewSearchExtension("GL_AMD_gcn_shader", extStart, extEnd);
#endif /* GL_AMD_gcn_shader */
#ifdef GL_AMD_gpu_shader_int64
  GLEW_AMD_gpu_shader_int64 = _glewSearchExtension("GL_AMD_gpu_shader_int64", extStart, extEnd);
#endif /* GL_AMD_gpu_shader_int64 */
#ifdef GL_AMD_interleaved_elements
  GLEW_AMD_interleaved_elements = _glewSearchExtension("GL_AMD_interleaved_elements", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_interleaved_elements) GLEW_AMD_interleaved_elements = !_glewInit_GL_AMD_interleaved_elements(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_interleaved_elements */
#ifdef GL_AMD_multi_draw_indirect
  GLEW_AMD_multi_draw_indirect = _glewSearchExtension("GL_AMD_multi_draw_indirect", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_multi_draw_indirect) GLEW_AMD_multi_draw_indirect = !_glewInit_GL_AMD_multi_draw_indirect(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_multi_draw_indirect */
#ifdef GL_AMD_name_gen_delete
  GLEW_AMD_name_gen_delete = _glewSearchExtension("GL_AMD_name_gen_delete", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_name_gen_delete) GLEW_AMD_name_gen_delete = !_glewInit_GL_AMD_name_gen_delete(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_name_gen_delete */
#ifdef GL_AMD_occlusion_query_event
  GLEW_AMD_occlusion_query_event = _glewSearchExtension("GL_AMD_occlusion_query_event", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_occlusion_query_event) GLEW_AMD_occlusion_query_event = !_glewInit_GL_AMD_occlusion_query_event(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_occlusion_query_event */
#ifdef GL_AMD_performance_monitor
  GLEW_AMD_performance_monitor = _glewSearchExtension("GL_AMD_performance_monitor", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_performance_monitor) GLEW_AMD_performance_monitor = !_glewInit_GL_AMD_performance_monitor(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_performance_monitor */
#ifdef GL_AMD_pinned_memory
  GLEW_AMD_pinned_memory = _glewSearchExtension("GL_AMD_pinned_memory", extStart, extEnd);
#endif /* GL_AMD_pinned_memory */
#ifdef GL_AMD_query_buffer_object
  GLEW_AMD_query_buffer_object = _glewSearchExtension("GL_AMD_query_buffer_object", extStart, extEnd);
#endif /* GL_AMD_query_buffer_object */
#ifdef GL_AMD_sample_positions
  GLEW_AMD_sample_positions = _glewSearchExtension("GL_AMD_sample_positions", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_sample_positions) GLEW_AMD_sample_positions = !_glewInit_GL_AMD_sample_positions(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_sample_positions */
#ifdef GL_AMD_seamless_cubemap_per_texture
  GLEW_AMD_seamless_cubemap_per_texture = _glewSearchExtension("GL_AMD_seamless_cubemap_per_texture", extStart, extEnd);
#endif /* GL_AMD_seamless_cubemap_per_texture */
#ifdef GL_AMD_shader_atomic_counter_ops
  GLEW_AMD_shader_atomic_counter_ops = _glewSearchExtension("GL_AMD_shader_atomic_counter_ops", extStart, extEnd);
#endif /* GL_AMD_shader_atomic_counter_ops */
#ifdef GL_AMD_shader_stencil_export
  GLEW_AMD_shader_stencil_export = _glewSearchExtension("GL_AMD_shader_stencil_export", extStart, extEnd);
#endif /* GL_AMD_shader_stencil_export */
#ifdef GL_AMD_shader_stencil_value_export
  GLEW_AMD_shader_stencil_value_export = _glewSearchExtension("GL_AMD_shader_stencil_value_export", extStart, extEnd);
#endif /* GL_AMD_shader_stencil_value_export */
#ifdef GL_AMD_shader_trinary_minmax
  GLEW_AMD_shader_trinary_minmax = _glewSearchExtension("GL_AMD_shader_trinary_minmax", extStart, extEnd);
#endif /* GL_AMD_shader_trinary_minmax */
#ifdef GL_AMD_sparse_texture
  GLEW_AMD_sparse_texture = _glewSearchExtension("GL_AMD_sparse_texture", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_sparse_texture) GLEW_AMD_sparse_texture = !_glewInit_GL_AMD_sparse_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_sparse_texture */
#ifdef GL_AMD_stencil_operation_extended
  GLEW_AMD_stencil_operation_extended = _glewSearchExtension("GL_AMD_stencil_operation_extended", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_stencil_operation_extended) GLEW_AMD_stencil_operation_extended = !_glewInit_GL_AMD_stencil_operation_extended(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_stencil_operation_extended */
#ifdef GL_AMD_texture_texture4
  GLEW_AMD_texture_texture4 = _glewSearchExtension("GL_AMD_texture_texture4", extStart, extEnd);
#endif /* GL_AMD_texture_texture4 */
#ifdef GL_AMD_transform_feedback3_lines_triangles
  GLEW_AMD_transform_feedback3_lines_triangles = _glewSearchExtension("GL_AMD_transform_feedback3_lines_triangles", extStart, extEnd);
#endif /* GL_AMD_transform_feedback3_lines_triangles */
#ifdef GL_AMD_transform_feedback4
  GLEW_AMD_transform_feedback4 = _glewSearchExtension("GL_AMD_transform_feedback4", extStart, extEnd);
#endif /* GL_AMD_transform_feedback4 */
#ifdef GL_AMD_vertex_shader_layer
  GLEW_AMD_vertex_shader_layer = _glewSearchExtension("GL_AMD_vertex_shader_layer", extStart, extEnd);
#endif /* GL_AMD_vertex_shader_layer */
#ifdef GL_AMD_vertex_shader_tessellator
  GLEW_AMD_vertex_shader_tessellator = _glewSearchExtension("GL_AMD_vertex_shader_tessellator", extStart, extEnd);
  if (glewExperimental || GLEW_AMD_vertex_shader_tessellator) GLEW_AMD_vertex_shader_tessellator = !_glewInit_GL_AMD_vertex_shader_tessellator(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_AMD_vertex_shader_tessellator */
#ifdef GL_AMD_vertex_shader_viewport_index
  GLEW_AMD_vertex_shader_viewport_index = _glewSearchExtension("GL_AMD_vertex_shader_viewport_index", extStart, extEnd);
#endif /* GL_AMD_vertex_shader_viewport_index */
#ifdef GL_ANGLE_depth_texture
  GLEW_ANGLE_depth_texture = _glewSearchExtension("GL_ANGLE_depth_texture", extStart, extEnd);
#endif /* GL_ANGLE_depth_texture */
#ifdef GL_ANGLE_framebuffer_blit
  GLEW_ANGLE_framebuffer_blit = _glewSearchExtension("GL_ANGLE_framebuffer_blit", extStart, extEnd);
  if (glewExperimental || GLEW_ANGLE_framebuffer_blit) GLEW_ANGLE_framebuffer_blit = !_glewInit_GL_ANGLE_framebuffer_blit(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ANGLE_framebuffer_blit */
#ifdef GL_ANGLE_framebuffer_multisample
  GLEW_ANGLE_framebuffer_multisample = _glewSearchExtension("GL_ANGLE_framebuffer_multisample", extStart, extEnd);
  if (glewExperimental || GLEW_ANGLE_framebuffer_multisample) GLEW_ANGLE_framebuffer_multisample = !_glewInit_GL_ANGLE_framebuffer_multisample(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ANGLE_framebuffer_multisample */
#ifdef GL_ANGLE_instanced_arrays
  GLEW_ANGLE_instanced_arrays = _glewSearchExtension("GL_ANGLE_instanced_arrays", extStart, extEnd);
  if (glewExperimental || GLEW_ANGLE_instanced_arrays) GLEW_ANGLE_instanced_arrays = !_glewInit_GL_ANGLE_instanced_arrays(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ANGLE_instanced_arrays */
#ifdef GL_ANGLE_pack_reverse_row_order
  GLEW_ANGLE_pack_reverse_row_order = _glewSearchExtension("GL_ANGLE_pack_reverse_row_order", extStart, extEnd);
#endif /* GL_ANGLE_pack_reverse_row_order */
#ifdef GL_ANGLE_program_binary
  GLEW_ANGLE_program_binary = _glewSearchExtension("GL_ANGLE_program_binary", extStart, extEnd);
#endif /* GL_ANGLE_program_binary */
#ifdef GL_ANGLE_texture_compression_dxt1
  GLEW_ANGLE_texture_compression_dxt1 = _glewSearchExtension("GL_ANGLE_texture_compression_dxt1", extStart, extEnd);
#endif /* GL_ANGLE_texture_compression_dxt1 */
#ifdef GL_ANGLE_texture_compression_dxt3
  GLEW_ANGLE_texture_compression_dxt3 = _glewSearchExtension("GL_ANGLE_texture_compression_dxt3", extStart, extEnd);
#endif /* GL_ANGLE_texture_compression_dxt3 */
#ifdef GL_ANGLE_texture_compression_dxt5
  GLEW_ANGLE_texture_compression_dxt5 = _glewSearchExtension("GL_ANGLE_texture_compression_dxt5", extStart, extEnd);
#endif /* GL_ANGLE_texture_compression_dxt5 */
#ifdef GL_ANGLE_texture_usage
  GLEW_ANGLE_texture_usage = _glewSearchExtension("GL_ANGLE_texture_usage", extStart, extEnd);
#endif /* GL_ANGLE_texture_usage */
#ifdef GL_ANGLE_timer_query
  GLEW_ANGLE_timer_query = _glewSearchExtension("GL_ANGLE_timer_query", extStart, extEnd);
  if (glewExperimental || GLEW_ANGLE_timer_query) GLEW_ANGLE_timer_query = !_glewInit_GL_ANGLE_timer_query(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ANGLE_timer_query */
#ifdef GL_ANGLE_translated_shader_source
  GLEW_ANGLE_translated_shader_source = _glewSearchExtension("GL_ANGLE_translated_shader_source", extStart, extEnd);
  if (glewExperimental || GLEW_ANGLE_translated_shader_source) GLEW_ANGLE_translated_shader_source = !_glewInit_GL_ANGLE_translated_shader_source(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ANGLE_translated_shader_source */
#ifdef GL_APPLE_aux_depth_stencil
  GLEW_APPLE_aux_depth_stencil = _glewSearchExtension("GL_APPLE_aux_depth_stencil", extStart, extEnd);
#endif /* GL_APPLE_aux_depth_stencil */
#ifdef GL_APPLE_client_storage
  GLEW_APPLE_client_storage = _glewSearchExtension("GL_APPLE_client_storage", extStart, extEnd);
#endif /* GL_APPLE_client_storage */
#ifdef GL_APPLE_element_array
  GLEW_APPLE_element_array = _glewSearchExtension("GL_APPLE_element_array", extStart, extEnd);
  if (glewExperimental || GLEW_APPLE_element_array) GLEW_APPLE_element_array = !_glewInit_GL_APPLE_element_array(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_APPLE_element_array */
#ifdef GL_APPLE_fence
  GLEW_APPLE_fence = _glewSearchExtension("GL_APPLE_fence", extStart, extEnd);
  if (glewExperimental || GLEW_APPLE_fence) GLEW_APPLE_fence = !_glewInit_GL_APPLE_fence(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_APPLE_fence */
#ifdef GL_APPLE_float_pixels
  GLEW_APPLE_float_pixels = _glewSearchExtension("GL_APPLE_float_pixels", extStart, extEnd);
#endif /* GL_APPLE_float_pixels */
#ifdef GL_APPLE_flush_buffer_range
  GLEW_APPLE_flush_buffer_range = _glewSearchExtension("GL_APPLE_flush_buffer_range", extStart, extEnd);
  if (glewExperimental || GLEW_APPLE_flush_buffer_range) GLEW_APPLE_flush_buffer_range = !_glewInit_GL_APPLE_flush_buffer_range(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_APPLE_flush_buffer_range */
#ifdef GL_APPLE_object_purgeable
  GLEW_APPLE_object_purgeable = _glewSearchExtension("GL_APPLE_object_purgeable", extStart, extEnd);
  if (glewExperimental || GLEW_APPLE_object_purgeable) GLEW_APPLE_object_purgeable = !_glewInit_GL_APPLE_object_purgeable(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_APPLE_object_purgeable */
#ifdef GL_APPLE_pixel_buffer
  GLEW_APPLE_pixel_buffer = _glewSearchExtension("GL_APPLE_pixel_buffer", extStart, extEnd);
#endif /* GL_APPLE_pixel_buffer */
#ifdef GL_APPLE_rgb_422
  GLEW_APPLE_rgb_422 = _glewSearchExtension("GL_APPLE_rgb_422", extStart, extEnd);
#endif /* GL_APPLE_rgb_422 */
#ifdef GL_APPLE_row_bytes
  GLEW_APPLE_row_bytes = _glewSearchExtension("GL_APPLE_row_bytes", extStart, extEnd);
#endif /* GL_APPLE_row_bytes */
#ifdef GL_APPLE_specular_vector
  GLEW_APPLE_specular_vector = _glewSearchExtension("GL_APPLE_specular_vector", extStart, extEnd);
#endif /* GL_APPLE_specular_vector */
#ifdef GL_APPLE_texture_range
  GLEW_APPLE_texture_range = _glewSearchExtension("GL_APPLE_texture_range", extStart, extEnd);
  if (glewExperimental || GLEW_APPLE_texture_range) GLEW_APPLE_texture_range = !_glewInit_GL_APPLE_texture_range(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_APPLE_texture_range */
#ifdef GL_APPLE_transform_hint
  GLEW_APPLE_transform_hint = _glewSearchExtension("GL_APPLE_transform_hint", extStart, extEnd);
#endif /* GL_APPLE_transform_hint */
#ifdef GL_APPLE_vertex_array_object
  GLEW_APPLE_vertex_array_object = _glewSearchExtension("GL_APPLE_vertex_array_object", extStart, extEnd);
  if (glewExperimental || GLEW_APPLE_vertex_array_object) GLEW_APPLE_vertex_array_object = !_glewInit_GL_APPLE_vertex_array_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_APPLE_vertex_array_object */
#ifdef GL_APPLE_vertex_array_range
  GLEW_APPLE_vertex_array_range = _glewSearchExtension("GL_APPLE_vertex_array_range", extStart, extEnd);
  if (glewExperimental || GLEW_APPLE_vertex_array_range) GLEW_APPLE_vertex_array_range = !_glewInit_GL_APPLE_vertex_array_range(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_APPLE_vertex_array_range */
#ifdef GL_APPLE_vertex_program_evaluators
  GLEW_APPLE_vertex_program_evaluators = _glewSearchExtension("GL_APPLE_vertex_program_evaluators", extStart, extEnd);
  if (glewExperimental || GLEW_APPLE_vertex_program_evaluators) GLEW_APPLE_vertex_program_evaluators = !_glewInit_GL_APPLE_vertex_program_evaluators(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_APPLE_vertex_program_evaluators */
#ifdef GL_APPLE_ycbcr_422
  GLEW_APPLE_ycbcr_422 = _glewSearchExtension("GL_APPLE_ycbcr_422", extStart, extEnd);
#endif /* GL_APPLE_ycbcr_422 */
#ifdef GL_ARB_ES2_compatibility
  GLEW_ARB_ES2_compatibility = _glewSearchExtension("GL_ARB_ES2_compatibility", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_ES2_compatibility) GLEW_ARB_ES2_compatibility = !_glewInit_GL_ARB_ES2_compatibility(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_ES2_compatibility */
#ifdef GL_ARB_ES3_1_compatibility
  GLEW_ARB_ES3_1_compatibility = _glewSearchExtension("GL_ARB_ES3_1_compatibility", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_ES3_1_compatibility) GLEW_ARB_ES3_1_compatibility = !_glewInit_GL_ARB_ES3_1_compatibility(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_ES3_1_compatibility */
#ifdef GL_ARB_ES3_2_compatibility
  GLEW_ARB_ES3_2_compatibility = _glewSearchExtension("GL_ARB_ES3_2_compatibility", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_ES3_2_compatibility) GLEW_ARB_ES3_2_compatibility = !_glewInit_GL_ARB_ES3_2_compatibility(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_ES3_2_compatibility */
#ifdef GL_ARB_ES3_compatibility
  GLEW_ARB_ES3_compatibility = _glewSearchExtension("GL_ARB_ES3_compatibility", extStart, extEnd);
#endif /* GL_ARB_ES3_compatibility */
#ifdef GL_ARB_arrays_of_arrays
  GLEW_ARB_arrays_of_arrays = _glewSearchExtension("GL_ARB_arrays_of_arrays", extStart, extEnd);
#endif /* GL_ARB_arrays_of_arrays */
#ifdef GL_ARB_base_instance
  GLEW_ARB_base_instance = _glewSearchExtension("GL_ARB_base_instance", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_base_instance) GLEW_ARB_base_instance = !_glewInit_GL_ARB_base_instance(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_base_instance */
#ifdef GL_ARB_bindless_texture
  GLEW_ARB_bindless_texture = _glewSearchExtension("GL_ARB_bindless_texture", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_bindless_texture) GLEW_ARB_bindless_texture = !_glewInit_GL_ARB_bindless_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_bindless_texture */
#ifdef GL_ARB_blend_func_extended
  GLEW_ARB_blend_func_extended = _glewSearchExtension("GL_ARB_blend_func_extended", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_blend_func_extended) GLEW_ARB_blend_func_extended = !_glewInit_GL_ARB_blend_func_extended(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_blend_func_extended */
#ifdef GL_ARB_buffer_storage
  GLEW_ARB_buffer_storage = _glewSearchExtension("GL_ARB_buffer_storage", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_buffer_storage) GLEW_ARB_buffer_storage = !_glewInit_GL_ARB_buffer_storage(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_buffer_storage */
#ifdef GL_ARB_cl_event
  GLEW_ARB_cl_event = _glewSearchExtension("GL_ARB_cl_event", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_cl_event) GLEW_ARB_cl_event = !_glewInit_GL_ARB_cl_event(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_cl_event */
#ifdef GL_ARB_clear_buffer_object
  GLEW_ARB_clear_buffer_object = _glewSearchExtension("GL_ARB_clear_buffer_object", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_clear_buffer_object) GLEW_ARB_clear_buffer_object = !_glewInit_GL_ARB_clear_buffer_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_clear_buffer_object */
#ifdef GL_ARB_clear_texture
  GLEW_ARB_clear_texture = _glewSearchExtension("GL_ARB_clear_texture", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_clear_texture) GLEW_ARB_clear_texture = !_glewInit_GL_ARB_clear_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_clear_texture */
#ifdef GL_ARB_clip_control
  GLEW_ARB_clip_control = _glewSearchExtension("GL_ARB_clip_control", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_clip_control) GLEW_ARB_clip_control = !_glewInit_GL_ARB_clip_control(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_clip_control */
#ifdef GL_ARB_color_buffer_float
  GLEW_ARB_color_buffer_float = _glewSearchExtension("GL_ARB_color_buffer_float", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_color_buffer_float) GLEW_ARB_color_buffer_float = !_glewInit_GL_ARB_color_buffer_float(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_color_buffer_float */
#ifdef GL_ARB_compatibility
  GLEW_ARB_compatibility = _glewSearchExtension("GL_ARB_compatibility", extStart, extEnd);
#endif /* GL_ARB_compatibility */
#ifdef GL_ARB_compressed_texture_pixel_storage
  GLEW_ARB_compressed_texture_pixel_storage = _glewSearchExtension("GL_ARB_compressed_texture_pixel_storage", extStart, extEnd);
#endif /* GL_ARB_compressed_texture_pixel_storage */
#ifdef GL_ARB_compute_shader
  GLEW_ARB_compute_shader = _glewSearchExtension("GL_ARB_compute_shader", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_compute_shader) GLEW_ARB_compute_shader = !_glewInit_GL_ARB_compute_shader(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_compute_shader */
#ifdef GL_ARB_compute_variable_group_size
  GLEW_ARB_compute_variable_group_size = _glewSearchExtension("GL_ARB_compute_variable_group_size", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_compute_variable_group_size) GLEW_ARB_compute_variable_group_size = !_glewInit_GL_ARB_compute_variable_group_size(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_compute_variable_group_size */
#ifdef GL_ARB_conditional_render_inverted
  GLEW_ARB_conditional_render_inverted = _glewSearchExtension("GL_ARB_conditional_render_inverted", extStart, extEnd);
#endif /* GL_ARB_conditional_render_inverted */
#ifdef GL_ARB_conservative_depth
  GLEW_ARB_conservative_depth = _glewSearchExtension("GL_ARB_conservative_depth", extStart, extEnd);
#endif /* GL_ARB_conservative_depth */
#ifdef GL_ARB_copy_buffer
  GLEW_ARB_copy_buffer = _glewSearchExtension("GL_ARB_copy_buffer", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_copy_buffer) GLEW_ARB_copy_buffer = !_glewInit_GL_ARB_copy_buffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_copy_buffer */
#ifdef GL_ARB_copy_image
  GLEW_ARB_copy_image = _glewSearchExtension("GL_ARB_copy_image", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_copy_image) GLEW_ARB_copy_image = !_glewInit_GL_ARB_copy_image(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_copy_image */
#ifdef GL_ARB_cull_distance
  GLEW_ARB_cull_distance = _glewSearchExtension("GL_ARB_cull_distance", extStart, extEnd);
#endif /* GL_ARB_cull_distance */
#ifdef GL_ARB_debug_output
  GLEW_ARB_debug_output = _glewSearchExtension("GL_ARB_debug_output", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_debug_output) GLEW_ARB_debug_output = !_glewInit_GL_ARB_debug_output(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_debug_output */
#ifdef GL_ARB_depth_buffer_float
  GLEW_ARB_depth_buffer_float = _glewSearchExtension("GL_ARB_depth_buffer_float", extStart, extEnd);
#endif /* GL_ARB_depth_buffer_float */
#ifdef GL_ARB_depth_clamp
  GLEW_ARB_depth_clamp = _glewSearchExtension("GL_ARB_depth_clamp", extStart, extEnd);
#endif /* GL_ARB_depth_clamp */
#ifdef GL_ARB_depth_texture
  GLEW_ARB_depth_texture = _glewSearchExtension("GL_ARB_depth_texture", extStart, extEnd);
#endif /* GL_ARB_depth_texture */
#ifdef GL_ARB_derivative_control
  GLEW_ARB_derivative_control = _glewSearchExtension("GL_ARB_derivative_control", extStart, extEnd);
#endif /* GL_ARB_derivative_control */
#ifdef GL_ARB_direct_state_access
  GLEW_ARB_direct_state_access = _glewSearchExtension("GL_ARB_direct_state_access", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_direct_state_access) GLEW_ARB_direct_state_access = !_glewInit_GL_ARB_direct_state_access(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_direct_state_access */
#ifdef GL_ARB_draw_buffers
  GLEW_ARB_draw_buffers = _glewSearchExtension("GL_ARB_draw_buffers", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_draw_buffers) GLEW_ARB_draw_buffers = !_glewInit_GL_ARB_draw_buffers(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_draw_buffers */
#ifdef GL_ARB_draw_buffers_blend
  GLEW_ARB_draw_buffers_blend = _glewSearchExtension("GL_ARB_draw_buffers_blend", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_draw_buffers_blend) GLEW_ARB_draw_buffers_blend = !_glewInit_GL_ARB_draw_buffers_blend(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_draw_buffers_blend */
#ifdef GL_ARB_draw_elements_base_vertex
  GLEW_ARB_draw_elements_base_vertex = _glewSearchExtension("GL_ARB_draw_elements_base_vertex", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_draw_elements_base_vertex) GLEW_ARB_draw_elements_base_vertex = !_glewInit_GL_ARB_draw_elements_base_vertex(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_draw_elements_base_vertex */
#ifdef GL_ARB_draw_indirect
  GLEW_ARB_draw_indirect = _glewSearchExtension("GL_ARB_draw_indirect", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_draw_indirect) GLEW_ARB_draw_indirect = !_glewInit_GL_ARB_draw_indirect(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_draw_indirect */
#ifdef GL_ARB_draw_instanced
  GLEW_ARB_draw_instanced = _glewSearchExtension("GL_ARB_draw_instanced", extStart, extEnd);
#endif /* GL_ARB_draw_instanced */
#ifdef GL_ARB_enhanced_layouts
  GLEW_ARB_enhanced_layouts = _glewSearchExtension("GL_ARB_enhanced_layouts", extStart, extEnd);
#endif /* GL_ARB_enhanced_layouts */
#ifdef GL_ARB_explicit_attrib_location
  GLEW_ARB_explicit_attrib_location = _glewSearchExtension("GL_ARB_explicit_attrib_location", extStart, extEnd);
#endif /* GL_ARB_explicit_attrib_location */
#ifdef GL_ARB_explicit_uniform_location
  GLEW_ARB_explicit_uniform_location = _glewSearchExtension("GL_ARB_explicit_uniform_location", extStart, extEnd);
#endif /* GL_ARB_explicit_uniform_location */
#ifdef GL_ARB_fragment_coord_conventions
  GLEW_ARB_fragment_coord_conventions = _glewSearchExtension("GL_ARB_fragment_coord_conventions", extStart, extEnd);
#endif /* GL_ARB_fragment_coord_conventions */
#ifdef GL_ARB_fragment_layer_viewport
  GLEW_ARB_fragment_layer_viewport = _glewSearchExtension("GL_ARB_fragment_layer_viewport", extStart, extEnd);
#endif /* GL_ARB_fragment_layer_viewport */
#ifdef GL_ARB_fragment_program
  GLEW_ARB_fragment_program = _glewSearchExtension("GL_ARB_fragment_program", extStart, extEnd);
#endif /* GL_ARB_fragment_program */
#ifdef GL_ARB_fragment_program_shadow
  GLEW_ARB_fragment_program_shadow = _glewSearchExtension("GL_ARB_fragment_program_shadow", extStart, extEnd);
#endif /* GL_ARB_fragment_program_shadow */
#ifdef GL_ARB_fragment_shader
  GLEW_ARB_fragment_shader = _glewSearchExtension("GL_ARB_fragment_shader", extStart, extEnd);
#endif /* GL_ARB_fragment_shader */
#ifdef GL_ARB_fragment_shader_interlock
  GLEW_ARB_fragment_shader_interlock = _glewSearchExtension("GL_ARB_fragment_shader_interlock", extStart, extEnd);
#endif /* GL_ARB_fragment_shader_interlock */
#ifdef GL_ARB_framebuffer_no_attachments
  GLEW_ARB_framebuffer_no_attachments = _glewSearchExtension("GL_ARB_framebuffer_no_attachments", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_framebuffer_no_attachments) GLEW_ARB_framebuffer_no_attachments = !_glewInit_GL_ARB_framebuffer_no_attachments(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_framebuffer_no_attachments */
#ifdef GL_ARB_framebuffer_object
  GLEW_ARB_framebuffer_object = _glewSearchExtension("GL_ARB_framebuffer_object", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_framebuffer_object) GLEW_ARB_framebuffer_object = !_glewInit_GL_ARB_framebuffer_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_framebuffer_object */
#ifdef GL_ARB_framebuffer_sRGB
  GLEW_ARB_framebuffer_sRGB = _glewSearchExtension("GL_ARB_framebuffer_sRGB", extStart, extEnd);
#endif /* GL_ARB_framebuffer_sRGB */
#ifdef GL_ARB_geometry_shader4
  GLEW_ARB_geometry_shader4 = _glewSearchExtension("GL_ARB_geometry_shader4", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_geometry_shader4) GLEW_ARB_geometry_shader4 = !_glewInit_GL_ARB_geometry_shader4(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_geometry_shader4 */
#ifdef GL_ARB_get_program_binary
  GLEW_ARB_get_program_binary = _glewSearchExtension("GL_ARB_get_program_binary", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_get_program_binary) GLEW_ARB_get_program_binary = !_glewInit_GL_ARB_get_program_binary(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_get_program_binary */
#ifdef GL_ARB_get_texture_sub_image
  GLEW_ARB_get_texture_sub_image = _glewSearchExtension("GL_ARB_get_texture_sub_image", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_get_texture_sub_image) GLEW_ARB_get_texture_sub_image = !_glewInit_GL_ARB_get_texture_sub_image(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_get_texture_sub_image */
#ifdef GL_ARB_gpu_shader5
  GLEW_ARB_gpu_shader5 = _glewSearchExtension("GL_ARB_gpu_shader5", extStart, extEnd);
#endif /* GL_ARB_gpu_shader5 */
#ifdef GL_ARB_gpu_shader_fp64
  GLEW_ARB_gpu_shader_fp64 = _glewSearchExtension("GL_ARB_gpu_shader_fp64", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_gpu_shader_fp64) GLEW_ARB_gpu_shader_fp64 = !_glewInit_GL_ARB_gpu_shader_fp64(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_gpu_shader_fp64 */
#ifdef GL_ARB_gpu_shader_int64
  GLEW_ARB_gpu_shader_int64 = _glewSearchExtension("GL_ARB_gpu_shader_int64", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_gpu_shader_int64) GLEW_ARB_gpu_shader_int64 = !_glewInit_GL_ARB_gpu_shader_int64(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_gpu_shader_int64 */
#ifdef GL_ARB_half_float_pixel
  GLEW_ARB_half_float_pixel = _glewSearchExtension("GL_ARB_half_float_pixel", extStart, extEnd);
#endif /* GL_ARB_half_float_pixel */
#ifdef GL_ARB_half_float_vertex
  GLEW_ARB_half_float_vertex = _glewSearchExtension("GL_ARB_half_float_vertex", extStart, extEnd);
#endif /* GL_ARB_half_float_vertex */
#ifdef GL_ARB_imaging
  GLEW_ARB_imaging = _glewSearchExtension("GL_ARB_imaging", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_imaging) GLEW_ARB_imaging = !_glewInit_GL_ARB_imaging(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_imaging */
#ifdef GL_ARB_indirect_parameters
  GLEW_ARB_indirect_parameters = _glewSearchExtension("GL_ARB_indirect_parameters", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_indirect_parameters) GLEW_ARB_indirect_parameters = !_glewInit_GL_ARB_indirect_parameters(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_indirect_parameters */
#ifdef GL_ARB_instanced_arrays
  GLEW_ARB_instanced_arrays = _glewSearchExtension("GL_ARB_instanced_arrays", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_instanced_arrays) GLEW_ARB_instanced_arrays = !_glewInit_GL_ARB_instanced_arrays(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_instanced_arrays */
#ifdef GL_ARB_internalformat_query
  GLEW_ARB_internalformat_query = _glewSearchExtension("GL_ARB_internalformat_query", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_internalformat_query) GLEW_ARB_internalformat_query = !_glewInit_GL_ARB_internalformat_query(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_internalformat_query */
#ifdef GL_ARB_internalformat_query2
  GLEW_ARB_internalformat_query2 = _glewSearchExtension("GL_ARB_internalformat_query2", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_internalformat_query2) GLEW_ARB_internalformat_query2 = !_glewInit_GL_ARB_internalformat_query2(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_internalformat_query2 */
#ifdef GL_ARB_invalidate_subdata
  GLEW_ARB_invalidate_subdata = _glewSearchExtension("GL_ARB_invalidate_subdata", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_invalidate_subdata) GLEW_ARB_invalidate_subdata = !_glewInit_GL_ARB_invalidate_subdata(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_invalidate_subdata */
#ifdef GL_ARB_map_buffer_alignment
  GLEW_ARB_map_buffer_alignment = _glewSearchExtension("GL_ARB_map_buffer_alignment", extStart, extEnd);
#endif /* GL_ARB_map_buffer_alignment */
#ifdef GL_ARB_map_buffer_range
  GLEW_ARB_map_buffer_range = _glewSearchExtension("GL_ARB_map_buffer_range", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_map_buffer_range) GLEW_ARB_map_buffer_range = !_glewInit_GL_ARB_map_buffer_range(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_map_buffer_range */
#ifdef GL_ARB_matrix_palette
  GLEW_ARB_matrix_palette = _glewSearchExtension("GL_ARB_matrix_palette", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_matrix_palette) GLEW_ARB_matrix_palette = !_glewInit_GL_ARB_matrix_palette(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_matrix_palette */
#ifdef GL_ARB_multi_bind
  GLEW_ARB_multi_bind = _glewSearchExtension("GL_ARB_multi_bind", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_multi_bind) GLEW_ARB_multi_bind = !_glewInit_GL_ARB_multi_bind(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_multi_bind */
#ifdef GL_ARB_multi_draw_indirect
  GLEW_ARB_multi_draw_indirect = _glewSearchExtension("GL_ARB_multi_draw_indirect", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_multi_draw_indirect) GLEW_ARB_multi_draw_indirect = !_glewInit_GL_ARB_multi_draw_indirect(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_multi_draw_indirect */
#ifdef GL_ARB_multisample
  GLEW_ARB_multisample = _glewSearchExtension("GL_ARB_multisample", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_multisample) GLEW_ARB_multisample = !_glewInit_GL_ARB_multisample(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_multisample */
#ifdef GL_ARB_multitexture
  GLEW_ARB_multitexture = _glewSearchExtension("GL_ARB_multitexture", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_multitexture) GLEW_ARB_multitexture = !_glewInit_GL_ARB_multitexture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_multitexture */
#ifdef GL_ARB_occlusion_query
  GLEW_ARB_occlusion_query = _glewSearchExtension("GL_ARB_occlusion_query", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_occlusion_query) GLEW_ARB_occlusion_query = !_glewInit_GL_ARB_occlusion_query(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_occlusion_query */
#ifdef GL_ARB_occlusion_query2
  GLEW_ARB_occlusion_query2 = _glewSearchExtension("GL_ARB_occlusion_query2", extStart, extEnd);
#endif /* GL_ARB_occlusion_query2 */
#ifdef GL_ARB_parallel_shader_compile
  GLEW_ARB_parallel_shader_compile = _glewSearchExtension("GL_ARB_parallel_shader_compile", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_parallel_shader_compile) GLEW_ARB_parallel_shader_compile = !_glewInit_GL_ARB_parallel_shader_compile(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_parallel_shader_compile */
#ifdef GL_ARB_pipeline_statistics_query
  GLEW_ARB_pipeline_statistics_query = _glewSearchExtension("GL_ARB_pipeline_statistics_query", extStart, extEnd);
#endif /* GL_ARB_pipeline_statistics_query */
#ifdef GL_ARB_pixel_buffer_object
  GLEW_ARB_pixel_buffer_object = _glewSearchExtension("GL_ARB_pixel_buffer_object", extStart, extEnd);
#endif /* GL_ARB_pixel_buffer_object */
#ifdef GL_ARB_point_parameters
  GLEW_ARB_point_parameters = _glewSearchExtension("GL_ARB_point_parameters", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_point_parameters) GLEW_ARB_point_parameters = !_glewInit_GL_ARB_point_parameters(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_point_parameters */
#ifdef GL_ARB_point_sprite
  GLEW_ARB_point_sprite = _glewSearchExtension("GL_ARB_point_sprite", extStart, extEnd);
#endif /* GL_ARB_point_sprite */
#ifdef GL_ARB_post_depth_coverage
  GLEW_ARB_post_depth_coverage = _glewSearchExtension("GL_ARB_post_depth_coverage", extStart, extEnd);
#endif /* GL_ARB_post_depth_coverage */
#ifdef GL_ARB_program_interface_query
  GLEW_ARB_program_interface_query = _glewSearchExtension("GL_ARB_program_interface_query", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_program_interface_query) GLEW_ARB_program_interface_query = !_glewInit_GL_ARB_program_interface_query(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_program_interface_query */
#ifdef GL_ARB_provoking_vertex
  GLEW_ARB_provoking_vertex = _glewSearchExtension("GL_ARB_provoking_vertex", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_provoking_vertex) GLEW_ARB_provoking_vertex = !_glewInit_GL_ARB_provoking_vertex(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_provoking_vertex */
#ifdef GL_ARB_query_buffer_object
  GLEW_ARB_query_buffer_object = _glewSearchExtension("GL_ARB_query_buffer_object", extStart, extEnd);
#endif /* GL_ARB_query_buffer_object */
#ifdef GL_ARB_robust_buffer_access_behavior
  GLEW_ARB_robust_buffer_access_behavior = _glewSearchExtension("GL_ARB_robust_buffer_access_behavior", extStart, extEnd);
#endif /* GL_ARB_robust_buffer_access_behavior */
#ifdef GL_ARB_robustness
  GLEW_ARB_robustness = _glewSearchExtension("GL_ARB_robustness", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_robustness) GLEW_ARB_robustness = !_glewInit_GL_ARB_robustness(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_robustness */
#ifdef GL_ARB_robustness_application_isolation
  GLEW_ARB_robustness_application_isolation = _glewSearchExtension("GL_ARB_robustness_application_isolation", extStart, extEnd);
#endif /* GL_ARB_robustness_application_isolation */
#ifdef GL_ARB_robustness_share_group_isolation
  GLEW_ARB_robustness_share_group_isolation = _glewSearchExtension("GL_ARB_robustness_share_group_isolation", extStart, extEnd);
#endif /* GL_ARB_robustness_share_group_isolation */
#ifdef GL_ARB_sample_locations
  GLEW_ARB_sample_locations = _glewSearchExtension("GL_ARB_sample_locations", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_sample_locations) GLEW_ARB_sample_locations = !_glewInit_GL_ARB_sample_locations(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_sample_locations */
#ifdef GL_ARB_sample_shading
  GLEW_ARB_sample_shading = _glewSearchExtension("GL_ARB_sample_shading", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_sample_shading) GLEW_ARB_sample_shading = !_glewInit_GL_ARB_sample_shading(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_sample_shading */
#ifdef GL_ARB_sampler_objects
  GLEW_ARB_sampler_objects = _glewSearchExtension("GL_ARB_sampler_objects", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_sampler_objects) GLEW_ARB_sampler_objects = !_glewInit_GL_ARB_sampler_objects(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_sampler_objects */
#ifdef GL_ARB_seamless_cube_map
  GLEW_ARB_seamless_cube_map = _glewSearchExtension("GL_ARB_seamless_cube_map", extStart, extEnd);
#endif /* GL_ARB_seamless_cube_map */
#ifdef GL_ARB_seamless_cubemap_per_texture
  GLEW_ARB_seamless_cubemap_per_texture = _glewSearchExtension("GL_ARB_seamless_cubemap_per_texture", extStart, extEnd);
#endif /* GL_ARB_seamless_cubemap_per_texture */
#ifdef GL_ARB_separate_shader_objects
  GLEW_ARB_separate_shader_objects = _glewSearchExtension("GL_ARB_separate_shader_objects", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_separate_shader_objects) GLEW_ARB_separate_shader_objects = !_glewInit_GL_ARB_separate_shader_objects(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_separate_shader_objects */
#ifdef GL_ARB_shader_atomic_counter_ops
  GLEW_ARB_shader_atomic_counter_ops = _glewSearchExtension("GL_ARB_shader_atomic_counter_ops", extStart, extEnd);
#endif /* GL_ARB_shader_atomic_counter_ops */
#ifdef GL_ARB_shader_atomic_counters
  GLEW_ARB_shader_atomic_counters = _glewSearchExtension("GL_ARB_shader_atomic_counters", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_shader_atomic_counters) GLEW_ARB_shader_atomic_counters = !_glewInit_GL_ARB_shader_atomic_counters(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_shader_atomic_counters */
#ifdef GL_ARB_shader_ballot
  GLEW_ARB_shader_ballot = _glewSearchExtension("GL_ARB_shader_ballot", extStart, extEnd);
#endif /* GL_ARB_shader_ballot */
#ifdef GL_ARB_shader_bit_encoding
  GLEW_ARB_shader_bit_encoding = _glewSearchExtension("GL_ARB_shader_bit_encoding", extStart, extEnd);
#endif /* GL_ARB_shader_bit_encoding */
#ifdef GL_ARB_shader_clock
  GLEW_ARB_shader_clock = _glewSearchExtension("GL_ARB_shader_clock", extStart, extEnd);
#endif /* GL_ARB_shader_clock */
#ifdef GL_ARB_shader_draw_parameters
  GLEW_ARB_shader_draw_parameters = _glewSearchExtension("GL_ARB_shader_draw_parameters", extStart, extEnd);
#endif /* GL_ARB_shader_draw_parameters */
#ifdef GL_ARB_shader_group_vote
  GLEW_ARB_shader_group_vote = _glewSearchExtension("GL_ARB_shader_group_vote", extStart, extEnd);
#endif /* GL_ARB_shader_group_vote */
#ifdef GL_ARB_shader_image_load_store
  GLEW_ARB_shader_image_load_store = _glewSearchExtension("GL_ARB_shader_image_load_store", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_shader_image_load_store) GLEW_ARB_shader_image_load_store = !_glewInit_GL_ARB_shader_image_load_store(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_shader_image_load_store */
#ifdef GL_ARB_shader_image_size
  GLEW_ARB_shader_image_size = _glewSearchExtension("GL_ARB_shader_image_size", extStart, extEnd);
#endif /* GL_ARB_shader_image_size */
#ifdef GL_ARB_shader_objects
  GLEW_ARB_shader_objects = _glewSearchExtension("GL_ARB_shader_objects", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_shader_objects) GLEW_ARB_shader_objects = !_glewInit_GL_ARB_shader_objects(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_shader_objects */
#ifdef GL_ARB_shader_precision
  GLEW_ARB_shader_precision = _glewSearchExtension("GL_ARB_shader_precision", extStart, extEnd);
#endif /* GL_ARB_shader_precision */
#ifdef GL_ARB_shader_stencil_export
  GLEW_ARB_shader_stencil_export = _glewSearchExtension("GL_ARB_shader_stencil_export", extStart, extEnd);
#endif /* GL_ARB_shader_stencil_export */
#ifdef GL_ARB_shader_storage_buffer_object
  GLEW_ARB_shader_storage_buffer_object = _glewSearchExtension("GL_ARB_shader_storage_buffer_object", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_shader_storage_buffer_object) GLEW_ARB_shader_storage_buffer_object = !_glewInit_GL_ARB_shader_storage_buffer_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_shader_storage_buffer_object */
#ifdef GL_ARB_shader_subroutine
  GLEW_ARB_shader_subroutine = _glewSearchExtension("GL_ARB_shader_subroutine", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_shader_subroutine) GLEW_ARB_shader_subroutine = !_glewInit_GL_ARB_shader_subroutine(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_shader_subroutine */
#ifdef GL_ARB_shader_texture_image_samples
  GLEW_ARB_shader_texture_image_samples = _glewSearchExtension("GL_ARB_shader_texture_image_samples", extStart, extEnd);
#endif /* GL_ARB_shader_texture_image_samples */
#ifdef GL_ARB_shader_texture_lod
  GLEW_ARB_shader_texture_lod = _glewSearchExtension("GL_ARB_shader_texture_lod", extStart, extEnd);
#endif /* GL_ARB_shader_texture_lod */
#ifdef GL_ARB_shader_viewport_layer_array
  GLEW_ARB_shader_viewport_layer_array = _glewSearchExtension("GL_ARB_shader_viewport_layer_array", extStart, extEnd);
#endif /* GL_ARB_shader_viewport_layer_array */
#ifdef GL_ARB_shading_language_100
  GLEW_ARB_shading_language_100 = _glewSearchExtension("GL_ARB_shading_language_100", extStart, extEnd);
#endif /* GL_ARB_shading_language_100 */
#ifdef GL_ARB_shading_language_420pack
  GLEW_ARB_shading_language_420pack = _glewSearchExtension("GL_ARB_shading_language_420pack", extStart, extEnd);
#endif /* GL_ARB_shading_language_420pack */
#ifdef GL_ARB_shading_language_include
  GLEW_ARB_shading_language_include = _glewSearchExtension("GL_ARB_shading_language_include", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_shading_language_include) GLEW_ARB_shading_language_include = !_glewInit_GL_ARB_shading_language_include(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_shading_language_include */
#ifdef GL_ARB_shading_language_packing
  GLEW_ARB_shading_language_packing = _glewSearchExtension("GL_ARB_shading_language_packing", extStart, extEnd);
#endif /* GL_ARB_shading_language_packing */
#ifdef GL_ARB_shadow
  GLEW_ARB_shadow = _glewSearchExtension("GL_ARB_shadow", extStart, extEnd);
#endif /* GL_ARB_shadow */
#ifdef GL_ARB_shadow_ambient
  GLEW_ARB_shadow_ambient = _glewSearchExtension("GL_ARB_shadow_ambient", extStart, extEnd);
#endif /* GL_ARB_shadow_ambient */
#ifdef GL_ARB_sparse_buffer
  GLEW_ARB_sparse_buffer = _glewSearchExtension("GL_ARB_sparse_buffer", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_sparse_buffer) GLEW_ARB_sparse_buffer = !_glewInit_GL_ARB_sparse_buffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_sparse_buffer */
#ifdef GL_ARB_sparse_texture
  GLEW_ARB_sparse_texture = _glewSearchExtension("GL_ARB_sparse_texture", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_sparse_texture) GLEW_ARB_sparse_texture = !_glewInit_GL_ARB_sparse_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_sparse_texture */
#ifdef GL_ARB_sparse_texture2
  GLEW_ARB_sparse_texture2 = _glewSearchExtension("GL_ARB_sparse_texture2", extStart, extEnd);
#endif /* GL_ARB_sparse_texture2 */
#ifdef GL_ARB_sparse_texture_clamp
  GLEW_ARB_sparse_texture_clamp = _glewSearchExtension("GL_ARB_sparse_texture_clamp", extStart, extEnd);
#endif /* GL_ARB_sparse_texture_clamp */
#ifdef GL_ARB_stencil_texturing
  GLEW_ARB_stencil_texturing = _glewSearchExtension("GL_ARB_stencil_texturing", extStart, extEnd);
#endif /* GL_ARB_stencil_texturing */
#ifdef GL_ARB_sync
  GLEW_ARB_sync = _glewSearchExtension("GL_ARB_sync", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_sync) GLEW_ARB_sync = !_glewInit_GL_ARB_sync(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_sync */
#ifdef GL_ARB_tessellation_shader
  GLEW_ARB_tessellation_shader = _glewSearchExtension("GL_ARB_tessellation_shader", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_tessellation_shader) GLEW_ARB_tessellation_shader = !_glewInit_GL_ARB_tessellation_shader(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_tessellation_shader */
#ifdef GL_ARB_texture_barrier
  GLEW_ARB_texture_barrier = _glewSearchExtension("GL_ARB_texture_barrier", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_texture_barrier) GLEW_ARB_texture_barrier = !_glewInit_GL_ARB_texture_barrier(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_texture_barrier */
#ifdef GL_ARB_texture_border_clamp
  GLEW_ARB_texture_border_clamp = _glewSearchExtension("GL_ARB_texture_border_clamp", extStart, extEnd);
#endif /* GL_ARB_texture_border_clamp */
#ifdef GL_ARB_texture_buffer_object
  GLEW_ARB_texture_buffer_object = _glewSearchExtension("GL_ARB_texture_buffer_object", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_texture_buffer_object) GLEW_ARB_texture_buffer_object = !_glewInit_GL_ARB_texture_buffer_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_texture_buffer_object */
#ifdef GL_ARB_texture_buffer_object_rgb32
  GLEW_ARB_texture_buffer_object_rgb32 = _glewSearchExtension("GL_ARB_texture_buffer_object_rgb32", extStart, extEnd);
#endif /* GL_ARB_texture_buffer_object_rgb32 */
#ifdef GL_ARB_texture_buffer_range
  GLEW_ARB_texture_buffer_range = _glewSearchExtension("GL_ARB_texture_buffer_range", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_texture_buffer_range) GLEW_ARB_texture_buffer_range = !_glewInit_GL_ARB_texture_buffer_range(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_texture_buffer_range */
#ifdef GL_ARB_texture_compression
  GLEW_ARB_texture_compression = _glewSearchExtension("GL_ARB_texture_compression", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_texture_compression) GLEW_ARB_texture_compression = !_glewInit_GL_ARB_texture_compression(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_texture_compression */
#ifdef GL_ARB_texture_compression_bptc
  GLEW_ARB_texture_compression_bptc = _glewSearchExtension("GL_ARB_texture_compression_bptc", extStart, extEnd);
#endif /* GL_ARB_texture_compression_bptc */
#ifdef GL_ARB_texture_compression_rgtc
  GLEW_ARB_texture_compression_rgtc = _glewSearchExtension("GL_ARB_texture_compression_rgtc", extStart, extEnd);
#endif /* GL_ARB_texture_compression_rgtc */
#ifdef GL_ARB_texture_cube_map
  GLEW_ARB_texture_cube_map = _glewSearchExtension("GL_ARB_texture_cube_map", extStart, extEnd);
#endif /* GL_ARB_texture_cube_map */
#ifdef GL_ARB_texture_cube_map_array
  GLEW_ARB_texture_cube_map_array = _glewSearchExtension("GL_ARB_texture_cube_map_array", extStart, extEnd);
#endif /* GL_ARB_texture_cube_map_array */
#ifdef GL_ARB_texture_env_add
  GLEW_ARB_texture_env_add = _glewSearchExtension("GL_ARB_texture_env_add", extStart, extEnd);
#endif /* GL_ARB_texture_env_add */
#ifdef GL_ARB_texture_env_combine
  GLEW_ARB_texture_env_combine = _glewSearchExtension("GL_ARB_texture_env_combine", extStart, extEnd);
#endif /* GL_ARB_texture_env_combine */
#ifdef GL_ARB_texture_env_crossbar
  GLEW_ARB_texture_env_crossbar = _glewSearchExtension("GL_ARB_texture_env_crossbar", extStart, extEnd);
#endif /* GL_ARB_texture_env_crossbar */
#ifdef GL_ARB_texture_env_dot3
  GLEW_ARB_texture_env_dot3 = _glewSearchExtension("GL_ARB_texture_env_dot3", extStart, extEnd);
#endif /* GL_ARB_texture_env_dot3 */
#ifdef GL_ARB_texture_filter_minmax
  GLEW_ARB_texture_filter_minmax = _glewSearchExtension("GL_ARB_texture_filter_minmax", extStart, extEnd);
#endif /* GL_ARB_texture_filter_minmax */
#ifdef GL_ARB_texture_float
  GLEW_ARB_texture_float = _glewSearchExtension("GL_ARB_texture_float", extStart, extEnd);
#endif /* GL_ARB_texture_float */
#ifdef GL_ARB_texture_gather
  GLEW_ARB_texture_gather = _glewSearchExtension("GL_ARB_texture_gather", extStart, extEnd);
#endif /* GL_ARB_texture_gather */
#ifdef GL_ARB_texture_mirror_clamp_to_edge
  GLEW_ARB_texture_mirror_clamp_to_edge = _glewSearchExtension("GL_ARB_texture_mirror_clamp_to_edge", extStart, extEnd);
#endif /* GL_ARB_texture_mirror_clamp_to_edge */
#ifdef GL_ARB_texture_mirrored_repeat
  GLEW_ARB_texture_mirrored_repeat = _glewSearchExtension("GL_ARB_texture_mirrored_repeat", extStart, extEnd);
#endif /* GL_ARB_texture_mirrored_repeat */
#ifdef GL_ARB_texture_multisample
  GLEW_ARB_texture_multisample = _glewSearchExtension("GL_ARB_texture_multisample", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_texture_multisample) GLEW_ARB_texture_multisample = !_glewInit_GL_ARB_texture_multisample(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_texture_multisample */
#ifdef GL_ARB_texture_non_power_of_two
  GLEW_ARB_texture_non_power_of_two = _glewSearchExtension("GL_ARB_texture_non_power_of_two", extStart, extEnd);
#endif /* GL_ARB_texture_non_power_of_two */
#ifdef GL_ARB_texture_query_levels
  GLEW_ARB_texture_query_levels = _glewSearchExtension("GL_ARB_texture_query_levels", extStart, extEnd);
#endif /* GL_ARB_texture_query_levels */
#ifdef GL_ARB_texture_query_lod
  GLEW_ARB_texture_query_lod = _glewSearchExtension("GL_ARB_texture_query_lod", extStart, extEnd);
#endif /* GL_ARB_texture_query_lod */
#ifdef GL_ARB_texture_rectangle
  GLEW_ARB_texture_rectangle = _glewSearchExtension("GL_ARB_texture_rectangle", extStart, extEnd);
#endif /* GL_ARB_texture_rectangle */
#ifdef GL_ARB_texture_rg
  GLEW_ARB_texture_rg = _glewSearchExtension("GL_ARB_texture_rg", extStart, extEnd);
#endif /* GL_ARB_texture_rg */
#ifdef GL_ARB_texture_rgb10_a2ui
  GLEW_ARB_texture_rgb10_a2ui = _glewSearchExtension("GL_ARB_texture_rgb10_a2ui", extStart, extEnd);
#endif /* GL_ARB_texture_rgb10_a2ui */
#ifdef GL_ARB_texture_stencil8
  GLEW_ARB_texture_stencil8 = _glewSearchExtension("GL_ARB_texture_stencil8", extStart, extEnd);
#endif /* GL_ARB_texture_stencil8 */
#ifdef GL_ARB_texture_storage
  GLEW_ARB_texture_storage = _glewSearchExtension("GL_ARB_texture_storage", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_texture_storage) GLEW_ARB_texture_storage = !_glewInit_GL_ARB_texture_storage(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_texture_storage */
#ifdef GL_ARB_texture_storage_multisample
  GLEW_ARB_texture_storage_multisample = _glewSearchExtension("GL_ARB_texture_storage_multisample", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_texture_storage_multisample) GLEW_ARB_texture_storage_multisample = !_glewInit_GL_ARB_texture_storage_multisample(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_texture_storage_multisample */
#ifdef GL_ARB_texture_swizzle
  GLEW_ARB_texture_swizzle = _glewSearchExtension("GL_ARB_texture_swizzle", extStart, extEnd);
#endif /* GL_ARB_texture_swizzle */
#ifdef GL_ARB_texture_view
  GLEW_ARB_texture_view = _glewSearchExtension("GL_ARB_texture_view", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_texture_view) GLEW_ARB_texture_view = !_glewInit_GL_ARB_texture_view(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_texture_view */
#ifdef GL_ARB_timer_query
  GLEW_ARB_timer_query = _glewSearchExtension("GL_ARB_timer_query", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_timer_query) GLEW_ARB_timer_query = !_glewInit_GL_ARB_timer_query(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_timer_query */
#ifdef GL_ARB_transform_feedback2
  GLEW_ARB_transform_feedback2 = _glewSearchExtension("GL_ARB_transform_feedback2", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_transform_feedback2) GLEW_ARB_transform_feedback2 = !_glewInit_GL_ARB_transform_feedback2(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_transform_feedback2 */
#ifdef GL_ARB_transform_feedback3
  GLEW_ARB_transform_feedback3 = _glewSearchExtension("GL_ARB_transform_feedback3", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_transform_feedback3) GLEW_ARB_transform_feedback3 = !_glewInit_GL_ARB_transform_feedback3(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_transform_feedback3 */
#ifdef GL_ARB_transform_feedback_instanced
  GLEW_ARB_transform_feedback_instanced = _glewSearchExtension("GL_ARB_transform_feedback_instanced", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_transform_feedback_instanced) GLEW_ARB_transform_feedback_instanced = !_glewInit_GL_ARB_transform_feedback_instanced(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_transform_feedback_instanced */
#ifdef GL_ARB_transform_feedback_overflow_query
  GLEW_ARB_transform_feedback_overflow_query = _glewSearchExtension("GL_ARB_transform_feedback_overflow_query", extStart, extEnd);
#endif /* GL_ARB_transform_feedback_overflow_query */
#ifdef GL_ARB_transpose_matrix
  GLEW_ARB_transpose_matrix = _glewSearchExtension("GL_ARB_transpose_matrix", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_transpose_matrix) GLEW_ARB_transpose_matrix = !_glewInit_GL_ARB_transpose_matrix(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_transpose_matrix */
#ifdef GL_ARB_uniform_buffer_object
  GLEW_ARB_uniform_buffer_object = _glewSearchExtension("GL_ARB_uniform_buffer_object", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_uniform_buffer_object) GLEW_ARB_uniform_buffer_object = !_glewInit_GL_ARB_uniform_buffer_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_uniform_buffer_object */
#ifdef GL_ARB_vertex_array_bgra
  GLEW_ARB_vertex_array_bgra = _glewSearchExtension("GL_ARB_vertex_array_bgra", extStart, extEnd);
#endif /* GL_ARB_vertex_array_bgra */
#ifdef GL_ARB_vertex_array_object
  GLEW_ARB_vertex_array_object = _glewSearchExtension("GL_ARB_vertex_array_object", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_vertex_array_object) GLEW_ARB_vertex_array_object = !_glewInit_GL_ARB_vertex_array_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_vertex_array_object */
#ifdef GL_ARB_vertex_attrib_64bit
  GLEW_ARB_vertex_attrib_64bit = _glewSearchExtension("GL_ARB_vertex_attrib_64bit", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_vertex_attrib_64bit) GLEW_ARB_vertex_attrib_64bit = !_glewInit_GL_ARB_vertex_attrib_64bit(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_vertex_attrib_64bit */
#ifdef GL_ARB_vertex_attrib_binding
  GLEW_ARB_vertex_attrib_binding = _glewSearchExtension("GL_ARB_vertex_attrib_binding", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_vertex_attrib_binding) GLEW_ARB_vertex_attrib_binding = !_glewInit_GL_ARB_vertex_attrib_binding(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_vertex_attrib_binding */
#ifdef GL_ARB_vertex_blend
  GLEW_ARB_vertex_blend = _glewSearchExtension("GL_ARB_vertex_blend", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_vertex_blend) GLEW_ARB_vertex_blend = !_glewInit_GL_ARB_vertex_blend(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_vertex_blend */
#ifdef GL_ARB_vertex_buffer_object
  GLEW_ARB_vertex_buffer_object = _glewSearchExtension("GL_ARB_vertex_buffer_object", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_vertex_buffer_object) GLEW_ARB_vertex_buffer_object = !_glewInit_GL_ARB_vertex_buffer_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_vertex_buffer_object */
#ifdef GL_ARB_vertex_program
  GLEW_ARB_vertex_program = _glewSearchExtension("GL_ARB_vertex_program", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_vertex_program) GLEW_ARB_vertex_program = !_glewInit_GL_ARB_vertex_program(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_vertex_program */
#ifdef GL_ARB_vertex_shader
  GLEW_ARB_vertex_shader = _glewSearchExtension("GL_ARB_vertex_shader", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_vertex_shader) { GLEW_ARB_vertex_shader = !_glewInit_GL_ARB_vertex_shader(GLEW_CONTEXT_ARG_VAR_INIT); _glewInit_GL_ARB_vertex_program(GLEW_CONTEXT_ARG_VAR_INIT); }
#endif /* GL_ARB_vertex_shader */
#ifdef GL_ARB_vertex_type_10f_11f_11f_rev
  GLEW_ARB_vertex_type_10f_11f_11f_rev = _glewSearchExtension("GL_ARB_vertex_type_10f_11f_11f_rev", extStart, extEnd);
#endif /* GL_ARB_vertex_type_10f_11f_11f_rev */
#ifdef GL_ARB_vertex_type_2_10_10_10_rev
  GLEW_ARB_vertex_type_2_10_10_10_rev = _glewSearchExtension("GL_ARB_vertex_type_2_10_10_10_rev", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_vertex_type_2_10_10_10_rev) GLEW_ARB_vertex_type_2_10_10_10_rev = !_glewInit_GL_ARB_vertex_type_2_10_10_10_rev(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_vertex_type_2_10_10_10_rev */
#ifdef GL_ARB_viewport_array
  GLEW_ARB_viewport_array = _glewSearchExtension("GL_ARB_viewport_array", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_viewport_array) GLEW_ARB_viewport_array = !_glewInit_GL_ARB_viewport_array(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_viewport_array */
#ifdef GL_ARB_window_pos
  GLEW_ARB_window_pos = _glewSearchExtension("GL_ARB_window_pos", extStart, extEnd);
  if (glewExperimental || GLEW_ARB_window_pos) GLEW_ARB_window_pos = !_glewInit_GL_ARB_window_pos(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ARB_window_pos */
#ifdef GL_ATIX_point_sprites
  GLEW_ATIX_point_sprites = _glewSearchExtension("GL_ATIX_point_sprites", extStart, extEnd);
#endif /* GL_ATIX_point_sprites */
#ifdef GL_ATIX_texture_env_combine3
  GLEW_ATIX_texture_env_combine3 = _glewSearchExtension("GL_ATIX_texture_env_combine3", extStart, extEnd);
#endif /* GL_ATIX_texture_env_combine3 */
#ifdef GL_ATIX_texture_env_route
  GLEW_ATIX_texture_env_route = _glewSearchExtension("GL_ATIX_texture_env_route", extStart, extEnd);
#endif /* GL_ATIX_texture_env_route */
#ifdef GL_ATIX_vertex_shader_output_point_size
  GLEW_ATIX_vertex_shader_output_point_size = _glewSearchExtension("GL_ATIX_vertex_shader_output_point_size", extStart, extEnd);
#endif /* GL_ATIX_vertex_shader_output_point_size */
#ifdef GL_ATI_draw_buffers
  GLEW_ATI_draw_buffers = _glewSearchExtension("GL_ATI_draw_buffers", extStart, extEnd);
  if (glewExperimental || GLEW_ATI_draw_buffers) GLEW_ATI_draw_buffers = !_glewInit_GL_ATI_draw_buffers(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ATI_draw_buffers */
#ifdef GL_ATI_element_array
  GLEW_ATI_element_array = _glewSearchExtension("GL_ATI_element_array", extStart, extEnd);
  if (glewExperimental || GLEW_ATI_element_array) GLEW_ATI_element_array = !_glewInit_GL_ATI_element_array(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ATI_element_array */
#ifdef GL_ATI_envmap_bumpmap
  GLEW_ATI_envmap_bumpmap = _glewSearchExtension("GL_ATI_envmap_bumpmap", extStart, extEnd);
  if (glewExperimental || GLEW_ATI_envmap_bumpmap) GLEW_ATI_envmap_bumpmap = !_glewInit_GL_ATI_envmap_bumpmap(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ATI_envmap_bumpmap */
#ifdef GL_ATI_fragment_shader
  GLEW_ATI_fragment_shader = _glewSearchExtension("GL_ATI_fragment_shader", extStart, extEnd);
  if (glewExperimental || GLEW_ATI_fragment_shader) GLEW_ATI_fragment_shader = !_glewInit_GL_ATI_fragment_shader(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ATI_fragment_shader */
#ifdef GL_ATI_map_object_buffer
  GLEW_ATI_map_object_buffer = _glewSearchExtension("GL_ATI_map_object_buffer", extStart, extEnd);
  if (glewExperimental || GLEW_ATI_map_object_buffer) GLEW_ATI_map_object_buffer = !_glewInit_GL_ATI_map_object_buffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ATI_map_object_buffer */
#ifdef GL_ATI_meminfo
  GLEW_ATI_meminfo = _glewSearchExtension("GL_ATI_meminfo", extStart, extEnd);
#endif /* GL_ATI_meminfo */
#ifdef GL_ATI_pn_triangles
  GLEW_ATI_pn_triangles = _glewSearchExtension("GL_ATI_pn_triangles", extStart, extEnd);
  if (glewExperimental || GLEW_ATI_pn_triangles) GLEW_ATI_pn_triangles = !_glewInit_GL_ATI_pn_triangles(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ATI_pn_triangles */
#ifdef GL_ATI_separate_stencil
  GLEW_ATI_separate_stencil = _glewSearchExtension("GL_ATI_separate_stencil", extStart, extEnd);
  if (glewExperimental || GLEW_ATI_separate_stencil) GLEW_ATI_separate_stencil = !_glewInit_GL_ATI_separate_stencil(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ATI_separate_stencil */
#ifdef GL_ATI_shader_texture_lod
  GLEW_ATI_shader_texture_lod = _glewSearchExtension("GL_ATI_shader_texture_lod", extStart, extEnd);
#endif /* GL_ATI_shader_texture_lod */
#ifdef GL_ATI_text_fragment_shader
  GLEW_ATI_text_fragment_shader = _glewSearchExtension("GL_ATI_text_fragment_shader", extStart, extEnd);
#endif /* GL_ATI_text_fragment_shader */
#ifdef GL_ATI_texture_compression_3dc
  GLEW_ATI_texture_compression_3dc = _glewSearchExtension("GL_ATI_texture_compression_3dc", extStart, extEnd);
#endif /* GL_ATI_texture_compression_3dc */
#ifdef GL_ATI_texture_env_combine3
  GLEW_ATI_texture_env_combine3 = _glewSearchExtension("GL_ATI_texture_env_combine3", extStart, extEnd);
#endif /* GL_ATI_texture_env_combine3 */
#ifdef GL_ATI_texture_float
  GLEW_ATI_texture_float = _glewSearchExtension("GL_ATI_texture_float", extStart, extEnd);
#endif /* GL_ATI_texture_float */
#ifdef GL_ATI_texture_mirror_once
  GLEW_ATI_texture_mirror_once = _glewSearchExtension("GL_ATI_texture_mirror_once", extStart, extEnd);
#endif /* GL_ATI_texture_mirror_once */
#ifdef GL_ATI_vertex_array_object
  GLEW_ATI_vertex_array_object = _glewSearchExtension("GL_ATI_vertex_array_object", extStart, extEnd);
  if (glewExperimental || GLEW_ATI_vertex_array_object) GLEW_ATI_vertex_array_object = !_glewInit_GL_ATI_vertex_array_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ATI_vertex_array_object */
#ifdef GL_ATI_vertex_attrib_array_object
  GLEW_ATI_vertex_attrib_array_object = _glewSearchExtension("GL_ATI_vertex_attrib_array_object", extStart, extEnd);
  if (glewExperimental || GLEW_ATI_vertex_attrib_array_object) GLEW_ATI_vertex_attrib_array_object = !_glewInit_GL_ATI_vertex_attrib_array_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ATI_vertex_attrib_array_object */
#ifdef GL_ATI_vertex_streams
  GLEW_ATI_vertex_streams = _glewSearchExtension("GL_ATI_vertex_streams", extStart, extEnd);
  if (glewExperimental || GLEW_ATI_vertex_streams) GLEW_ATI_vertex_streams = !_glewInit_GL_ATI_vertex_streams(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_ATI_vertex_streams */
#ifdef GL_EXT_422_pixels
  GLEW_EXT_422_pixels = _glewSearchExtension("GL_EXT_422_pixels", extStart, extEnd);
#endif /* GL_EXT_422_pixels */
#ifdef GL_EXT_Cg_shader
  GLEW_EXT_Cg_shader = _glewSearchExtension("GL_EXT_Cg_shader", extStart, extEnd);
#endif /* GL_EXT_Cg_shader */
#ifdef GL_EXT_abgr
  GLEW_EXT_abgr = _glewSearchExtension("GL_EXT_abgr", extStart, extEnd);
#endif /* GL_EXT_abgr */
#ifdef GL_EXT_bgra
  GLEW_EXT_bgra = _glewSearchExtension("GL_EXT_bgra", extStart, extEnd);
#endif /* GL_EXT_bgra */
#ifdef GL_EXT_bindable_uniform
  GLEW_EXT_bindable_uniform = _glewSearchExtension("GL_EXT_bindable_uniform", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_bindable_uniform) GLEW_EXT_bindable_uniform = !_glewInit_GL_EXT_bindable_uniform(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_bindable_uniform */
#ifdef GL_EXT_blend_color
  GLEW_EXT_blend_color = _glewSearchExtension("GL_EXT_blend_color", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_blend_color) GLEW_EXT_blend_color = !_glewInit_GL_EXT_blend_color(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_blend_color */
#ifdef GL_EXT_blend_equation_separate
  GLEW_EXT_blend_equation_separate = _glewSearchExtension("GL_EXT_blend_equation_separate", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_blend_equation_separate) GLEW_EXT_blend_equation_separate = !_glewInit_GL_EXT_blend_equation_separate(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_blend_equation_separate */
#ifdef GL_EXT_blend_func_separate
  GLEW_EXT_blend_func_separate = _glewSearchExtension("GL_EXT_blend_func_separate", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_blend_func_separate) GLEW_EXT_blend_func_separate = !_glewInit_GL_EXT_blend_func_separate(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_blend_func_separate */
#ifdef GL_EXT_blend_logic_op
  GLEW_EXT_blend_logic_op = _glewSearchExtension("GL_EXT_blend_logic_op", extStart, extEnd);
#endif /* GL_EXT_blend_logic_op */
#ifdef GL_EXT_blend_minmax
  GLEW_EXT_blend_minmax = _glewSearchExtension("GL_EXT_blend_minmax", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_blend_minmax) GLEW_EXT_blend_minmax = !_glewInit_GL_EXT_blend_minmax(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_blend_minmax */
#ifdef GL_EXT_blend_subtract
  GLEW_EXT_blend_subtract = _glewSearchExtension("GL_EXT_blend_subtract", extStart, extEnd);
#endif /* GL_EXT_blend_subtract */
#ifdef GL_EXT_clip_volume_hint
  GLEW_EXT_clip_volume_hint = _glewSearchExtension("GL_EXT_clip_volume_hint", extStart, extEnd);
#endif /* GL_EXT_clip_volume_hint */
#ifdef GL_EXT_cmyka
  GLEW_EXT_cmyka = _glewSearchExtension("GL_EXT_cmyka", extStart, extEnd);
#endif /* GL_EXT_cmyka */
#ifdef GL_EXT_color_subtable
  GLEW_EXT_color_subtable = _glewSearchExtension("GL_EXT_color_subtable", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_color_subtable) GLEW_EXT_color_subtable = !_glewInit_GL_EXT_color_subtable(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_color_subtable */
#ifdef GL_EXT_compiled_vertex_array
  GLEW_EXT_compiled_vertex_array = _glewSearchExtension("GL_EXT_compiled_vertex_array", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_compiled_vertex_array) GLEW_EXT_compiled_vertex_array = !_glewInit_GL_EXT_compiled_vertex_array(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_compiled_vertex_array */
#ifdef GL_EXT_convolution
  GLEW_EXT_convolution = _glewSearchExtension("GL_EXT_convolution", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_convolution) GLEW_EXT_convolution = !_glewInit_GL_EXT_convolution(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_convolution */
#ifdef GL_EXT_coordinate_frame
  GLEW_EXT_coordinate_frame = _glewSearchExtension("GL_EXT_coordinate_frame", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_coordinate_frame) GLEW_EXT_coordinate_frame = !_glewInit_GL_EXT_coordinate_frame(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_coordinate_frame */
#ifdef GL_EXT_copy_texture
  GLEW_EXT_copy_texture = _glewSearchExtension("GL_EXT_copy_texture", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_copy_texture) GLEW_EXT_copy_texture = !_glewInit_GL_EXT_copy_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_copy_texture */
#ifdef GL_EXT_cull_vertex
  GLEW_EXT_cull_vertex = _glewSearchExtension("GL_EXT_cull_vertex", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_cull_vertex) GLEW_EXT_cull_vertex = !_glewInit_GL_EXT_cull_vertex(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_cull_vertex */
#ifdef GL_EXT_debug_label
  GLEW_EXT_debug_label = _glewSearchExtension("GL_EXT_debug_label", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_debug_label) GLEW_EXT_debug_label = !_glewInit_GL_EXT_debug_label(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_debug_label */
#ifdef GL_EXT_debug_marker
  GLEW_EXT_debug_marker = _glewSearchExtension("GL_EXT_debug_marker", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_debug_marker) GLEW_EXT_debug_marker = !_glewInit_GL_EXT_debug_marker(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_debug_marker */
#ifdef GL_EXT_depth_bounds_test
  GLEW_EXT_depth_bounds_test = _glewSearchExtension("GL_EXT_depth_bounds_test", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_depth_bounds_test) GLEW_EXT_depth_bounds_test = !_glewInit_GL_EXT_depth_bounds_test(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_depth_bounds_test */
#ifdef GL_EXT_direct_state_access
  GLEW_EXT_direct_state_access = _glewSearchExtension("GL_EXT_direct_state_access", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_direct_state_access) GLEW_EXT_direct_state_access = !_glewInit_GL_EXT_direct_state_access(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_direct_state_access */
#ifdef GL_EXT_draw_buffers2
  GLEW_EXT_draw_buffers2 = _glewSearchExtension("GL_EXT_draw_buffers2", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_draw_buffers2) GLEW_EXT_draw_buffers2 = !_glewInit_GL_EXT_draw_buffers2(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_draw_buffers2 */
#ifdef GL_EXT_draw_instanced
  GLEW_EXT_draw_instanced = _glewSearchExtension("GL_EXT_draw_instanced", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_draw_instanced) GLEW_EXT_draw_instanced = !_glewInit_GL_EXT_draw_instanced(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_draw_instanced */
#ifdef GL_EXT_draw_range_elements
  GLEW_EXT_draw_range_elements = _glewSearchExtension("GL_EXT_draw_range_elements", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_draw_range_elements) GLEW_EXT_draw_range_elements = !_glewInit_GL_EXT_draw_range_elements(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_draw_range_elements */
#ifdef GL_EXT_fog_coord
  GLEW_EXT_fog_coord = _glewSearchExtension("GL_EXT_fog_coord", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_fog_coord) GLEW_EXT_fog_coord = !_glewInit_GL_EXT_fog_coord(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_fog_coord */
#ifdef GL_EXT_fragment_lighting
  GLEW_EXT_fragment_lighting = _glewSearchExtension("GL_EXT_fragment_lighting", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_fragment_lighting) GLEW_EXT_fragment_lighting = !_glewInit_GL_EXT_fragment_lighting(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_fragment_lighting */
#ifdef GL_EXT_framebuffer_blit
  GLEW_EXT_framebuffer_blit = _glewSearchExtension("GL_EXT_framebuffer_blit", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_framebuffer_blit) GLEW_EXT_framebuffer_blit = !_glewInit_GL_EXT_framebuffer_blit(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_framebuffer_blit */
#ifdef GL_EXT_framebuffer_multisample
  GLEW_EXT_framebuffer_multisample = _glewSearchExtension("GL_EXT_framebuffer_multisample", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_framebuffer_multisample) GLEW_EXT_framebuffer_multisample = !_glewInit_GL_EXT_framebuffer_multisample(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_framebuffer_multisample */
#ifdef GL_EXT_framebuffer_multisample_blit_scaled
  GLEW_EXT_framebuffer_multisample_blit_scaled = _glewSearchExtension("GL_EXT_framebuffer_multisample_blit_scaled", extStart, extEnd);
#endif /* GL_EXT_framebuffer_multisample_blit_scaled */
#ifdef GL_EXT_framebuffer_object
  GLEW_EXT_framebuffer_object = _glewSearchExtension("GL_EXT_framebuffer_object", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_framebuffer_object) GLEW_EXT_framebuffer_object = !_glewInit_GL_EXT_framebuffer_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_framebuffer_object */
#ifdef GL_EXT_framebuffer_sRGB
  GLEW_EXT_framebuffer_sRGB = _glewSearchExtension("GL_EXT_framebuffer_sRGB", extStart, extEnd);
#endif /* GL_EXT_framebuffer_sRGB */
#ifdef GL_EXT_geometry_shader4
  GLEW_EXT_geometry_shader4 = _glewSearchExtension("GL_EXT_geometry_shader4", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_geometry_shader4) GLEW_EXT_geometry_shader4 = !_glewInit_GL_EXT_geometry_shader4(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_geometry_shader4 */
#ifdef GL_EXT_gpu_program_parameters
  GLEW_EXT_gpu_program_parameters = _glewSearchExtension("GL_EXT_gpu_program_parameters", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_gpu_program_parameters) GLEW_EXT_gpu_program_parameters = !_glewInit_GL_EXT_gpu_program_parameters(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_gpu_program_parameters */
#ifdef GL_EXT_gpu_shader4
  GLEW_EXT_gpu_shader4 = _glewSearchExtension("GL_EXT_gpu_shader4", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_gpu_shader4) GLEW_EXT_gpu_shader4 = !_glewInit_GL_EXT_gpu_shader4(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_gpu_shader4 */
#ifdef GL_EXT_histogram
  GLEW_EXT_histogram = _glewSearchExtension("GL_EXT_histogram", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_histogram) GLEW_EXT_histogram = !_glewInit_GL_EXT_histogram(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_histogram */
#ifdef GL_EXT_index_array_formats
  GLEW_EXT_index_array_formats = _glewSearchExtension("GL_EXT_index_array_formats", extStart, extEnd);
#endif /* GL_EXT_index_array_formats */
#ifdef GL_EXT_index_func
  GLEW_EXT_index_func = _glewSearchExtension("GL_EXT_index_func", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_index_func) GLEW_EXT_index_func = !_glewInit_GL_EXT_index_func(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_index_func */
#ifdef GL_EXT_index_material
  GLEW_EXT_index_material = _glewSearchExtension("GL_EXT_index_material", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_index_material) GLEW_EXT_index_material = !_glewInit_GL_EXT_index_material(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_index_material */
#ifdef GL_EXT_index_texture
  GLEW_EXT_index_texture = _glewSearchExtension("GL_EXT_index_texture", extStart, extEnd);
#endif /* GL_EXT_index_texture */
#ifdef GL_EXT_light_texture
  GLEW_EXT_light_texture = _glewSearchExtension("GL_EXT_light_texture", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_light_texture) GLEW_EXT_light_texture = !_glewInit_GL_EXT_light_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_light_texture */
#ifdef GL_EXT_misc_attribute
  GLEW_EXT_misc_attribute = _glewSearchExtension("GL_EXT_misc_attribute", extStart, extEnd);
#endif /* GL_EXT_misc_attribute */
#ifdef GL_EXT_multi_draw_arrays
  GLEW_EXT_multi_draw_arrays = _glewSearchExtension("GL_EXT_multi_draw_arrays", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_multi_draw_arrays) GLEW_EXT_multi_draw_arrays = !_glewInit_GL_EXT_multi_draw_arrays(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_multi_draw_arrays */
#ifdef GL_EXT_multisample
  GLEW_EXT_multisample = _glewSearchExtension("GL_EXT_multisample", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_multisample) GLEW_EXT_multisample = !_glewInit_GL_EXT_multisample(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_multisample */
#ifdef GL_EXT_packed_depth_stencil
  GLEW_EXT_packed_depth_stencil = _glewSearchExtension("GL_EXT_packed_depth_stencil", extStart, extEnd);
#endif /* GL_EXT_packed_depth_stencil */
#ifdef GL_EXT_packed_float
  GLEW_EXT_packed_float = _glewSearchExtension("GL_EXT_packed_float", extStart, extEnd);
#endif /* GL_EXT_packed_float */
#ifdef GL_EXT_packed_pixels
  GLEW_EXT_packed_pixels = _glewSearchExtension("GL_EXT_packed_pixels", extStart, extEnd);
#endif /* GL_EXT_packed_pixels */
#ifdef GL_EXT_paletted_texture
  GLEW_EXT_paletted_texture = _glewSearchExtension("GL_EXT_paletted_texture", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_paletted_texture) GLEW_EXT_paletted_texture = !_glewInit_GL_EXT_paletted_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_paletted_texture */
#ifdef GL_EXT_pixel_buffer_object
  GLEW_EXT_pixel_buffer_object = _glewSearchExtension("GL_EXT_pixel_buffer_object", extStart, extEnd);
#endif /* GL_EXT_pixel_buffer_object */
#ifdef GL_EXT_pixel_transform
  GLEW_EXT_pixel_transform = _glewSearchExtension("GL_EXT_pixel_transform", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_pixel_transform) GLEW_EXT_pixel_transform = !_glewInit_GL_EXT_pixel_transform(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_pixel_transform */
#ifdef GL_EXT_pixel_transform_color_table
  GLEW_EXT_pixel_transform_color_table = _glewSearchExtension("GL_EXT_pixel_transform_color_table", extStart, extEnd);
#endif /* GL_EXT_pixel_transform_color_table */
#ifdef GL_EXT_point_parameters
  GLEW_EXT_point_parameters = _glewSearchExtension("GL_EXT_point_parameters", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_point_parameters) GLEW_EXT_point_parameters = !_glewInit_GL_EXT_point_parameters(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_point_parameters */
#ifdef GL_EXT_polygon_offset
  GLEW_EXT_polygon_offset = _glewSearchExtension("GL_EXT_polygon_offset", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_polygon_offset) GLEW_EXT_polygon_offset = !_glewInit_GL_EXT_polygon_offset(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_polygon_offset */
#ifdef GL_EXT_polygon_offset_clamp
  GLEW_EXT_polygon_offset_clamp = _glewSearchExtension("GL_EXT_polygon_offset_clamp", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_polygon_offset_clamp) GLEW_EXT_polygon_offset_clamp = !_glewInit_GL_EXT_polygon_offset_clamp(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_polygon_offset_clamp */
#ifdef GL_EXT_post_depth_coverage
  GLEW_EXT_post_depth_coverage = _glewSearchExtension("GL_EXT_post_depth_coverage", extStart, extEnd);
#endif /* GL_EXT_post_depth_coverage */
#ifdef GL_EXT_provoking_vertex
  GLEW_EXT_provoking_vertex = _glewSearchExtension("GL_EXT_provoking_vertex", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_provoking_vertex) GLEW_EXT_provoking_vertex = !_glewInit_GL_EXT_provoking_vertex(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_provoking_vertex */
#ifdef GL_EXT_raster_multisample
  GLEW_EXT_raster_multisample = _glewSearchExtension("GL_EXT_raster_multisample", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_raster_multisample) GLEW_EXT_raster_multisample = !_glewInit_GL_EXT_raster_multisample(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_raster_multisample */
#ifdef GL_EXT_rescale_normal
  GLEW_EXT_rescale_normal = _glewSearchExtension("GL_EXT_rescale_normal", extStart, extEnd);
#endif /* GL_EXT_rescale_normal */
#ifdef GL_EXT_scene_marker
  GLEW_EXT_scene_marker = _glewSearchExtension("GL_EXT_scene_marker", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_scene_marker) GLEW_EXT_scene_marker = !_glewInit_GL_EXT_scene_marker(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_scene_marker */
#ifdef GL_EXT_secondary_color
  GLEW_EXT_secondary_color = _glewSearchExtension("GL_EXT_secondary_color", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_secondary_color) GLEW_EXT_secondary_color = !_glewInit_GL_EXT_secondary_color(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_secondary_color */
#ifdef GL_EXT_separate_shader_objects
  GLEW_EXT_separate_shader_objects = _glewSearchExtension("GL_EXT_separate_shader_objects", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_separate_shader_objects) GLEW_EXT_separate_shader_objects = !_glewInit_GL_EXT_separate_shader_objects(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_separate_shader_objects */
#ifdef GL_EXT_separate_specular_color
  GLEW_EXT_separate_specular_color = _glewSearchExtension("GL_EXT_separate_specular_color", extStart, extEnd);
#endif /* GL_EXT_separate_specular_color */
#ifdef GL_EXT_shader_image_load_formatted
  GLEW_EXT_shader_image_load_formatted = _glewSearchExtension("GL_EXT_shader_image_load_formatted", extStart, extEnd);
#endif /* GL_EXT_shader_image_load_formatted */
#ifdef GL_EXT_shader_image_load_store
  GLEW_EXT_shader_image_load_store = _glewSearchExtension("GL_EXT_shader_image_load_store", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_shader_image_load_store) GLEW_EXT_shader_image_load_store = !_glewInit_GL_EXT_shader_image_load_store(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_shader_image_load_store */
#ifdef GL_EXT_shader_integer_mix
  GLEW_EXT_shader_integer_mix = _glewSearchExtension("GL_EXT_shader_integer_mix", extStart, extEnd);
#endif /* GL_EXT_shader_integer_mix */
#ifdef GL_EXT_shadow_funcs
  GLEW_EXT_shadow_funcs = _glewSearchExtension("GL_EXT_shadow_funcs", extStart, extEnd);
#endif /* GL_EXT_shadow_funcs */
#ifdef GL_EXT_shared_texture_palette
  GLEW_EXT_shared_texture_palette = _glewSearchExtension("GL_EXT_shared_texture_palette", extStart, extEnd);
#endif /* GL_EXT_shared_texture_palette */
#ifdef GL_EXT_sparse_texture2
  GLEW_EXT_sparse_texture2 = _glewSearchExtension("GL_EXT_sparse_texture2", extStart, extEnd);
#endif /* GL_EXT_sparse_texture2 */
#ifdef GL_EXT_stencil_clear_tag
  GLEW_EXT_stencil_clear_tag = _glewSearchExtension("GL_EXT_stencil_clear_tag", extStart, extEnd);
#endif /* GL_EXT_stencil_clear_tag */
#ifdef GL_EXT_stencil_two_side
  GLEW_EXT_stencil_two_side = _glewSearchExtension("GL_EXT_stencil_two_side", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_stencil_two_side) GLEW_EXT_stencil_two_side = !_glewInit_GL_EXT_stencil_two_side(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_stencil_two_side */
#ifdef GL_EXT_stencil_wrap
  GLEW_EXT_stencil_wrap = _glewSearchExtension("GL_EXT_stencil_wrap", extStart, extEnd);
#endif /* GL_EXT_stencil_wrap */
#ifdef GL_EXT_subtexture
  GLEW_EXT_subtexture = _glewSearchExtension("GL_EXT_subtexture", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_subtexture) GLEW_EXT_subtexture = !_glewInit_GL_EXT_subtexture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_subtexture */
#ifdef GL_EXT_texture
  GLEW_EXT_texture = _glewSearchExtension("GL_EXT_texture", extStart, extEnd);
#endif /* GL_EXT_texture */
#ifdef GL_EXT_texture3D
  GLEW_EXT_texture3D = _glewSearchExtension("GL_EXT_texture3D", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_texture3D) GLEW_EXT_texture3D = !_glewInit_GL_EXT_texture3D(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_texture3D */
#ifdef GL_EXT_texture_array
  GLEW_EXT_texture_array = _glewSearchExtension("GL_EXT_texture_array", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_texture_array) GLEW_EXT_texture_array = !_glewInit_GL_EXT_texture_array(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_texture_array */
#ifdef GL_EXT_texture_buffer_object
  GLEW_EXT_texture_buffer_object = _glewSearchExtension("GL_EXT_texture_buffer_object", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_texture_buffer_object) GLEW_EXT_texture_buffer_object = !_glewInit_GL_EXT_texture_buffer_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_texture_buffer_object */
#ifdef GL_EXT_texture_compression_dxt1
  GLEW_EXT_texture_compression_dxt1 = _glewSearchExtension("GL_EXT_texture_compression_dxt1", extStart, extEnd);
#endif /* GL_EXT_texture_compression_dxt1 */
#ifdef GL_EXT_texture_compression_latc
  GLEW_EXT_texture_compression_latc = _glewSearchExtension("GL_EXT_texture_compression_latc", extStart, extEnd);
#endif /* GL_EXT_texture_compression_latc */
#ifdef GL_EXT_texture_compression_rgtc
  GLEW_EXT_texture_compression_rgtc = _glewSearchExtension("GL_EXT_texture_compression_rgtc", extStart, extEnd);
#endif /* GL_EXT_texture_compression_rgtc */
#ifdef GL_EXT_texture_compression_s3tc
  GLEW_EXT_texture_compression_s3tc = _glewSearchExtension("GL_EXT_texture_compression_s3tc", extStart, extEnd);
#endif /* GL_EXT_texture_compression_s3tc */
#ifdef GL_EXT_texture_cube_map
  GLEW_EXT_texture_cube_map = _glewSearchExtension("GL_EXT_texture_cube_map", extStart, extEnd);
#endif /* GL_EXT_texture_cube_map */
#ifdef GL_EXT_texture_edge_clamp
  GLEW_EXT_texture_edge_clamp = _glewSearchExtension("GL_EXT_texture_edge_clamp", extStart, extEnd);
#endif /* GL_EXT_texture_edge_clamp */
#ifdef GL_EXT_texture_env
  GLEW_EXT_texture_env = _glewSearchExtension("GL_EXT_texture_env", extStart, extEnd);
#endif /* GL_EXT_texture_env */
#ifdef GL_EXT_texture_env_add
  GLEW_EXT_texture_env_add = _glewSearchExtension("GL_EXT_texture_env_add", extStart, extEnd);
#endif /* GL_EXT_texture_env_add */
#ifdef GL_EXT_texture_env_combine
  GLEW_EXT_texture_env_combine = _glewSearchExtension("GL_EXT_texture_env_combine", extStart, extEnd);
#endif /* GL_EXT_texture_env_combine */
#ifdef GL_EXT_texture_env_dot3
  GLEW_EXT_texture_env_dot3 = _glewSearchExtension("GL_EXT_texture_env_dot3", extStart, extEnd);
#endif /* GL_EXT_texture_env_dot3 */
#ifdef GL_EXT_texture_filter_anisotropic
  GLEW_EXT_texture_filter_anisotropic = _glewSearchExtension("GL_EXT_texture_filter_anisotropic", extStart, extEnd);
#endif /* GL_EXT_texture_filter_anisotropic */
#ifdef GL_EXT_texture_filter_minmax
  GLEW_EXT_texture_filter_minmax = _glewSearchExtension("GL_EXT_texture_filter_minmax", extStart, extEnd);
#endif /* GL_EXT_texture_filter_minmax */
#ifdef GL_EXT_texture_integer
  GLEW_EXT_texture_integer = _glewSearchExtension("GL_EXT_texture_integer", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_texture_integer) GLEW_EXT_texture_integer = !_glewInit_GL_EXT_texture_integer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_texture_integer */
#ifdef GL_EXT_texture_lod_bias
  GLEW_EXT_texture_lod_bias = _glewSearchExtension("GL_EXT_texture_lod_bias", extStart, extEnd);
#endif /* GL_EXT_texture_lod_bias */
#ifdef GL_EXT_texture_mirror_clamp
  GLEW_EXT_texture_mirror_clamp = _glewSearchExtension("GL_EXT_texture_mirror_clamp", extStart, extEnd);
#endif /* GL_EXT_texture_mirror_clamp */
#ifdef GL_EXT_texture_object
  GLEW_EXT_texture_object = _glewSearchExtension("GL_EXT_texture_object", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_texture_object) GLEW_EXT_texture_object = !_glewInit_GL_EXT_texture_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_texture_object */
#ifdef GL_EXT_texture_perturb_normal
  GLEW_EXT_texture_perturb_normal = _glewSearchExtension("GL_EXT_texture_perturb_normal", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_texture_perturb_normal) GLEW_EXT_texture_perturb_normal = !_glewInit_GL_EXT_texture_perturb_normal(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_texture_perturb_normal */
#ifdef GL_EXT_texture_rectangle
  GLEW_EXT_texture_rectangle = _glewSearchExtension("GL_EXT_texture_rectangle", extStart, extEnd);
#endif /* GL_EXT_texture_rectangle */
#ifdef GL_EXT_texture_sRGB
  GLEW_EXT_texture_sRGB = _glewSearchExtension("GL_EXT_texture_sRGB", extStart, extEnd);
#endif /* GL_EXT_texture_sRGB */
#ifdef GL_EXT_texture_sRGB_decode
  GLEW_EXT_texture_sRGB_decode = _glewSearchExtension("GL_EXT_texture_sRGB_decode", extStart, extEnd);
#endif /* GL_EXT_texture_sRGB_decode */
#ifdef GL_EXT_texture_shared_exponent
  GLEW_EXT_texture_shared_exponent = _glewSearchExtension("GL_EXT_texture_shared_exponent", extStart, extEnd);
#endif /* GL_EXT_texture_shared_exponent */
#ifdef GL_EXT_texture_snorm
  GLEW_EXT_texture_snorm = _glewSearchExtension("GL_EXT_texture_snorm", extStart, extEnd);
#endif /* GL_EXT_texture_snorm */
#ifdef GL_EXT_texture_swizzle
  GLEW_EXT_texture_swizzle = _glewSearchExtension("GL_EXT_texture_swizzle", extStart, extEnd);
#endif /* GL_EXT_texture_swizzle */
#ifdef GL_EXT_timer_query
  GLEW_EXT_timer_query = _glewSearchExtension("GL_EXT_timer_query", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_timer_query) GLEW_EXT_timer_query = !_glewInit_GL_EXT_timer_query(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_timer_query */
#ifdef GL_EXT_transform_feedback
  GLEW_EXT_transform_feedback = _glewSearchExtension("GL_EXT_transform_feedback", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_transform_feedback) GLEW_EXT_transform_feedback = !_glewInit_GL_EXT_transform_feedback(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_transform_feedback */
#ifdef GL_EXT_vertex_array
  GLEW_EXT_vertex_array = _glewSearchExtension("GL_EXT_vertex_array", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_vertex_array) GLEW_EXT_vertex_array = !_glewInit_GL_EXT_vertex_array(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_vertex_array */
#ifdef GL_EXT_vertex_array_bgra
  GLEW_EXT_vertex_array_bgra = _glewSearchExtension("GL_EXT_vertex_array_bgra", extStart, extEnd);
#endif /* GL_EXT_vertex_array_bgra */
#ifdef GL_EXT_vertex_attrib_64bit
  GLEW_EXT_vertex_attrib_64bit = _glewSearchExtension("GL_EXT_vertex_attrib_64bit", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_vertex_attrib_64bit) GLEW_EXT_vertex_attrib_64bit = !_glewInit_GL_EXT_vertex_attrib_64bit(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_vertex_attrib_64bit */
#ifdef GL_EXT_vertex_shader
  GLEW_EXT_vertex_shader = _glewSearchExtension("GL_EXT_vertex_shader", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_vertex_shader) GLEW_EXT_vertex_shader = !_glewInit_GL_EXT_vertex_shader(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_vertex_shader */
#ifdef GL_EXT_vertex_weighting
  GLEW_EXT_vertex_weighting = _glewSearchExtension("GL_EXT_vertex_weighting", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_vertex_weighting) GLEW_EXT_vertex_weighting = !_glewInit_GL_EXT_vertex_weighting(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_vertex_weighting */
#ifdef GL_EXT_x11_sync_object
  GLEW_EXT_x11_sync_object = _glewSearchExtension("GL_EXT_x11_sync_object", extStart, extEnd);
  if (glewExperimental || GLEW_EXT_x11_sync_object) GLEW_EXT_x11_sync_object = !_glewInit_GL_EXT_x11_sync_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_EXT_x11_sync_object */
#ifdef GL_GREMEDY_frame_terminator
  GLEW_GREMEDY_frame_terminator = _glewSearchExtension("GL_GREMEDY_frame_terminator", extStart, extEnd);
  if (glewExperimental || GLEW_GREMEDY_frame_terminator) GLEW_GREMEDY_frame_terminator = !_glewInit_GL_GREMEDY_frame_terminator(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_GREMEDY_frame_terminator */
#ifdef GL_GREMEDY_string_marker
  GLEW_GREMEDY_string_marker = _glewSearchExtension("GL_GREMEDY_string_marker", extStart, extEnd);
  if (glewExperimental || GLEW_GREMEDY_string_marker) GLEW_GREMEDY_string_marker = !_glewInit_GL_GREMEDY_string_marker(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_GREMEDY_string_marker */
#ifdef GL_HP_convolution_border_modes
  GLEW_HP_convolution_border_modes = _glewSearchExtension("GL_HP_convolution_border_modes", extStart, extEnd);
#endif /* GL_HP_convolution_border_modes */
#ifdef GL_HP_image_transform
  GLEW_HP_image_transform = _glewSearchExtension("GL_HP_image_transform", extStart, extEnd);
  if (glewExperimental || GLEW_HP_image_transform) GLEW_HP_image_transform = !_glewInit_GL_HP_image_transform(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_HP_image_transform */
#ifdef GL_HP_occlusion_test
  GLEW_HP_occlusion_test = _glewSearchExtension("GL_HP_occlusion_test", extStart, extEnd);
#endif /* GL_HP_occlusion_test */
#ifdef GL_HP_texture_lighting
  GLEW_HP_texture_lighting = _glewSearchExtension("GL_HP_texture_lighting", extStart, extEnd);
#endif /* GL_HP_texture_lighting */
#ifdef GL_IBM_cull_vertex
  GLEW_IBM_cull_vertex = _glewSearchExtension("GL_IBM_cull_vertex", extStart, extEnd);
#endif /* GL_IBM_cull_vertex */
#ifdef GL_IBM_multimode_draw_arrays
  GLEW_IBM_multimode_draw_arrays = _glewSearchExtension("GL_IBM_multimode_draw_arrays", extStart, extEnd);
  if (glewExperimental || GLEW_IBM_multimode_draw_arrays) GLEW_IBM_multimode_draw_arrays = !_glewInit_GL_IBM_multimode_draw_arrays(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_IBM_multimode_draw_arrays */
#ifdef GL_IBM_rasterpos_clip
  GLEW_IBM_rasterpos_clip = _glewSearchExtension("GL_IBM_rasterpos_clip", extStart, extEnd);
#endif /* GL_IBM_rasterpos_clip */
#ifdef GL_IBM_static_data
  GLEW_IBM_static_data = _glewSearchExtension("GL_IBM_static_data", extStart, extEnd);
#endif /* GL_IBM_static_data */
#ifdef GL_IBM_texture_mirrored_repeat
  GLEW_IBM_texture_mirrored_repeat = _glewSearchExtension("GL_IBM_texture_mirrored_repeat", extStart, extEnd);
#endif /* GL_IBM_texture_mirrored_repeat */
#ifdef GL_IBM_vertex_array_lists
  GLEW_IBM_vertex_array_lists = _glewSearchExtension("GL_IBM_vertex_array_lists", extStart, extEnd);
  if (glewExperimental || GLEW_IBM_vertex_array_lists) GLEW_IBM_vertex_array_lists = !_glewInit_GL_IBM_vertex_array_lists(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_IBM_vertex_array_lists */
#ifdef GL_INGR_color_clamp
  GLEW_INGR_color_clamp = _glewSearchExtension("GL_INGR_color_clamp", extStart, extEnd);
#endif /* GL_INGR_color_clamp */
#ifdef GL_INGR_interlace_read
  GLEW_INGR_interlace_read = _glewSearchExtension("GL_INGR_interlace_read", extStart, extEnd);
#endif /* GL_INGR_interlace_read */
#ifdef GL_INTEL_fragment_shader_ordering
  GLEW_INTEL_fragment_shader_ordering = _glewSearchExtension("GL_INTEL_fragment_shader_ordering", extStart, extEnd);
#endif /* GL_INTEL_fragment_shader_ordering */
#ifdef GL_INTEL_framebuffer_CMAA
  GLEW_INTEL_framebuffer_CMAA = _glewSearchExtension("GL_INTEL_framebuffer_CMAA", extStart, extEnd);
#endif /* GL_INTEL_framebuffer_CMAA */
#ifdef GL_INTEL_map_texture
  GLEW_INTEL_map_texture = _glewSearchExtension("GL_INTEL_map_texture", extStart, extEnd);
  if (glewExperimental || GLEW_INTEL_map_texture) GLEW_INTEL_map_texture = !_glewInit_GL_INTEL_map_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_INTEL_map_texture */
#ifdef GL_INTEL_parallel_arrays
  GLEW_INTEL_parallel_arrays = _glewSearchExtension("GL_INTEL_parallel_arrays", extStart, extEnd);
  if (glewExperimental || GLEW_INTEL_parallel_arrays) GLEW_INTEL_parallel_arrays = !_glewInit_GL_INTEL_parallel_arrays(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_INTEL_parallel_arrays */
#ifdef GL_INTEL_performance_query
  GLEW_INTEL_performance_query = _glewSearchExtension("GL_INTEL_performance_query", extStart, extEnd);
  if (glewExperimental || GLEW_INTEL_performance_query) GLEW_INTEL_performance_query = !_glewInit_GL_INTEL_performance_query(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_INTEL_performance_query */
#ifdef GL_INTEL_texture_scissor
  GLEW_INTEL_texture_scissor = _glewSearchExtension("GL_INTEL_texture_scissor", extStart, extEnd);
  if (glewExperimental || GLEW_INTEL_texture_scissor) GLEW_INTEL_texture_scissor = !_glewInit_GL_INTEL_texture_scissor(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_INTEL_texture_scissor */
#ifdef GL_KHR_blend_equation_advanced
  GLEW_KHR_blend_equation_advanced = _glewSearchExtension("GL_KHR_blend_equation_advanced", extStart, extEnd);
  if (glewExperimental || GLEW_KHR_blend_equation_advanced) GLEW_KHR_blend_equation_advanced = !_glewInit_GL_KHR_blend_equation_advanced(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_KHR_blend_equation_advanced */
#ifdef GL_KHR_blend_equation_advanced_coherent
  GLEW_KHR_blend_equation_advanced_coherent = _glewSearchExtension("GL_KHR_blend_equation_advanced_coherent", extStart, extEnd);
#endif /* GL_KHR_blend_equation_advanced_coherent */
#ifdef GL_KHR_context_flush_control
  GLEW_KHR_context_flush_control = _glewSearchExtension("GL_KHR_context_flush_control", extStart, extEnd);
#endif /* GL_KHR_context_flush_control */
#ifdef GL_KHR_debug
  GLEW_KHR_debug = _glewSearchExtension("GL_KHR_debug", extStart, extEnd);
  if (glewExperimental || GLEW_KHR_debug) GLEW_KHR_debug = !_glewInit_GL_KHR_debug(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_KHR_debug */
#ifdef GL_KHR_no_error
  GLEW_KHR_no_error = _glewSearchExtension("GL_KHR_no_error", extStart, extEnd);
#endif /* GL_KHR_no_error */
#ifdef GL_KHR_robust_buffer_access_behavior
  GLEW_KHR_robust_buffer_access_behavior = _glewSearchExtension("GL_KHR_robust_buffer_access_behavior", extStart, extEnd);
#endif /* GL_KHR_robust_buffer_access_behavior */
#ifdef GL_KHR_robustness
  GLEW_KHR_robustness = _glewSearchExtension("GL_KHR_robustness", extStart, extEnd);
  if (glewExperimental || GLEW_KHR_robustness) GLEW_KHR_robustness = !_glewInit_GL_KHR_robustness(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_KHR_robustness */
#ifdef GL_KHR_texture_compression_astc_hdr
  GLEW_KHR_texture_compression_astc_hdr = _glewSearchExtension("GL_KHR_texture_compression_astc_hdr", extStart, extEnd);
#endif /* GL_KHR_texture_compression_astc_hdr */
#ifdef GL_KHR_texture_compression_astc_ldr
  GLEW_KHR_texture_compression_astc_ldr = _glewSearchExtension("GL_KHR_texture_compression_astc_ldr", extStart, extEnd);
#endif /* GL_KHR_texture_compression_astc_ldr */
#ifdef GL_KTX_buffer_region
  GLEW_KTX_buffer_region = _glewSearchExtension("GL_KTX_buffer_region", extStart, extEnd);
  if (glewExperimental || GLEW_KTX_buffer_region) GLEW_KTX_buffer_region = !_glewInit_GL_KTX_buffer_region(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_KTX_buffer_region */
#ifdef GL_MESAX_texture_stack
  GLEW_MESAX_texture_stack = _glewSearchExtension("GL_MESAX_texture_stack", extStart, extEnd);
#endif /* GL_MESAX_texture_stack */
#ifdef GL_MESA_pack_invert
  GLEW_MESA_pack_invert = _glewSearchExtension("GL_MESA_pack_invert", extStart, extEnd);
#endif /* GL_MESA_pack_invert */
#ifdef GL_MESA_resize_buffers
  GLEW_MESA_resize_buffers = _glewSearchExtension("GL_MESA_resize_buffers", extStart, extEnd);
  if (glewExperimental || GLEW_MESA_resize_buffers) GLEW_MESA_resize_buffers = !_glewInit_GL_MESA_resize_buffers(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_MESA_resize_buffers */
#ifdef GL_MESA_window_pos
  GLEW_MESA_window_pos = _glewSearchExtension("GL_MESA_window_pos", extStart, extEnd);
  if (glewExperimental || GLEW_MESA_window_pos) GLEW_MESA_window_pos = !_glewInit_GL_MESA_window_pos(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_MESA_window_pos */
#ifdef GL_MESA_ycbcr_texture
  GLEW_MESA_ycbcr_texture = _glewSearchExtension("GL_MESA_ycbcr_texture", extStart, extEnd);
#endif /* GL_MESA_ycbcr_texture */
#ifdef GL_NVX_conditional_render
  GLEW_NVX_conditional_render = _glewSearchExtension("GL_NVX_conditional_render", extStart, extEnd);
  if (glewExperimental || GLEW_NVX_conditional_render) GLEW_NVX_conditional_render = !_glewInit_GL_NVX_conditional_render(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NVX_conditional_render */
#ifdef GL_NVX_gpu_memory_info
  GLEW_NVX_gpu_memory_info = _glewSearchExtension("GL_NVX_gpu_memory_info", extStart, extEnd);
#endif /* GL_NVX_gpu_memory_info */
#ifdef GL_NV_bindless_multi_draw_indirect
  GLEW_NV_bindless_multi_draw_indirect = _glewSearchExtension("GL_NV_bindless_multi_draw_indirect", extStart, extEnd);
  if (glewExperimental || GLEW_NV_bindless_multi_draw_indirect) GLEW_NV_bindless_multi_draw_indirect = !_glewInit_GL_NV_bindless_multi_draw_indirect(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_bindless_multi_draw_indirect */
#ifdef GL_NV_bindless_multi_draw_indirect_count
  GLEW_NV_bindless_multi_draw_indirect_count = _glewSearchExtension("GL_NV_bindless_multi_draw_indirect_count", extStart, extEnd);
  if (glewExperimental || GLEW_NV_bindless_multi_draw_indirect_count) GLEW_NV_bindless_multi_draw_indirect_count = !_glewInit_GL_NV_bindless_multi_draw_indirect_count(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_bindless_multi_draw_indirect_count */
#ifdef GL_NV_bindless_texture
  GLEW_NV_bindless_texture = _glewSearchExtension("GL_NV_bindless_texture", extStart, extEnd);
  if (glewExperimental || GLEW_NV_bindless_texture) GLEW_NV_bindless_texture = !_glewInit_GL_NV_bindless_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_bindless_texture */
#ifdef GL_NV_blend_equation_advanced
  GLEW_NV_blend_equation_advanced = _glewSearchExtension("GL_NV_blend_equation_advanced", extStart, extEnd);
  if (glewExperimental || GLEW_NV_blend_equation_advanced) GLEW_NV_blend_equation_advanced = !_glewInit_GL_NV_blend_equation_advanced(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_blend_equation_advanced */
#ifdef GL_NV_blend_equation_advanced_coherent
  GLEW_NV_blend_equation_advanced_coherent = _glewSearchExtension("GL_NV_blend_equation_advanced_coherent", extStart, extEnd);
#endif /* GL_NV_blend_equation_advanced_coherent */
#ifdef GL_NV_blend_square
  GLEW_NV_blend_square = _glewSearchExtension("GL_NV_blend_square", extStart, extEnd);
#endif /* GL_NV_blend_square */
#ifdef GL_NV_compute_program5
  GLEW_NV_compute_program5 = _glewSearchExtension("GL_NV_compute_program5", extStart, extEnd);
#endif /* GL_NV_compute_program5 */
#ifdef GL_NV_conditional_render
  GLEW_NV_conditional_render = _glewSearchExtension("GL_NV_conditional_render", extStart, extEnd);
  if (glewExperimental || GLEW_NV_conditional_render) GLEW_NV_conditional_render = !_glewInit_GL_NV_conditional_render(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_conditional_render */
#ifdef GL_NV_conservative_raster
  GLEW_NV_conservative_raster = _glewSearchExtension("GL_NV_conservative_raster", extStart, extEnd);
  if (glewExperimental || GLEW_NV_conservative_raster) GLEW_NV_conservative_raster = !_glewInit_GL_NV_conservative_raster(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_conservative_raster */
#ifdef GL_NV_conservative_raster_dilate
  GLEW_NV_conservative_raster_dilate = _glewSearchExtension("GL_NV_conservative_raster_dilate", extStart, extEnd);
  if (glewExperimental || GLEW_NV_conservative_raster_dilate) GLEW_NV_conservative_raster_dilate = !_glewInit_GL_NV_conservative_raster_dilate(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_conservative_raster_dilate */
#ifdef GL_NV_copy_depth_to_color
  GLEW_NV_copy_depth_to_color = _glewSearchExtension("GL_NV_copy_depth_to_color", extStart, extEnd);
#endif /* GL_NV_copy_depth_to_color */
#ifdef GL_NV_copy_image
  GLEW_NV_copy_image = _glewSearchExtension("GL_NV_copy_image", extStart, extEnd);
  if (glewExperimental || GLEW_NV_copy_image) GLEW_NV_copy_image = !_glewInit_GL_NV_copy_image(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_copy_image */
#ifdef GL_NV_deep_texture3D
  GLEW_NV_deep_texture3D = _glewSearchExtension("GL_NV_deep_texture3D", extStart, extEnd);
#endif /* GL_NV_deep_texture3D */
#ifdef GL_NV_depth_buffer_float
  GLEW_NV_depth_buffer_float = _glewSearchExtension("GL_NV_depth_buffer_float", extStart, extEnd);
  if (glewExperimental || GLEW_NV_depth_buffer_float) GLEW_NV_depth_buffer_float = !_glewInit_GL_NV_depth_buffer_float(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_depth_buffer_float */
#ifdef GL_NV_depth_clamp
  GLEW_NV_depth_clamp = _glewSearchExtension("GL_NV_depth_clamp", extStart, extEnd);
#endif /* GL_NV_depth_clamp */
#ifdef GL_NV_depth_range_unclamped
  GLEW_NV_depth_range_unclamped = _glewSearchExtension("GL_NV_depth_range_unclamped", extStart, extEnd);
#endif /* GL_NV_depth_range_unclamped */
#ifdef GL_NV_draw_texture
  GLEW_NV_draw_texture = _glewSearchExtension("GL_NV_draw_texture", extStart, extEnd);
  if (glewExperimental || GLEW_NV_draw_texture) GLEW_NV_draw_texture = !_glewInit_GL_NV_draw_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_draw_texture */
#ifdef GL_NV_evaluators
  GLEW_NV_evaluators = _glewSearchExtension("GL_NV_evaluators", extStart, extEnd);
  if (glewExperimental || GLEW_NV_evaluators) GLEW_NV_evaluators = !_glewInit_GL_NV_evaluators(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_evaluators */
#ifdef GL_NV_explicit_multisample
  GLEW_NV_explicit_multisample = _glewSearchExtension("GL_NV_explicit_multisample", extStart, extEnd);
  if (glewExperimental || GLEW_NV_explicit_multisample) GLEW_NV_explicit_multisample = !_glewInit_GL_NV_explicit_multisample(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_explicit_multisample */
#ifdef GL_NV_fence
  GLEW_NV_fence = _glewSearchExtension("GL_NV_fence", extStart, extEnd);
  if (glewExperimental || GLEW_NV_fence) GLEW_NV_fence = !_glewInit_GL_NV_fence(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_fence */
#ifdef GL_NV_fill_rectangle
  GLEW_NV_fill_rectangle = _glewSearchExtension("GL_NV_fill_rectangle", extStart, extEnd);
#endif /* GL_NV_fill_rectangle */
#ifdef GL_NV_float_buffer
  GLEW_NV_float_buffer = _glewSearchExtension("GL_NV_float_buffer", extStart, extEnd);
#endif /* GL_NV_float_buffer */
#ifdef GL_NV_fog_distance
  GLEW_NV_fog_distance = _glewSearchExtension("GL_NV_fog_distance", extStart, extEnd);
#endif /* GL_NV_fog_distance */
#ifdef GL_NV_fragment_coverage_to_color
  GLEW_NV_fragment_coverage_to_color = _glewSearchExtension("GL_NV_fragment_coverage_to_color", extStart, extEnd);
  if (glewExperimental || GLEW_NV_fragment_coverage_to_color) GLEW_NV_fragment_coverage_to_color = !_glewInit_GL_NV_fragment_coverage_to_color(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_fragment_coverage_to_color */
#ifdef GL_NV_fragment_program
  GLEW_NV_fragment_program = _glewSearchExtension("GL_NV_fragment_program", extStart, extEnd);
  if (glewExperimental || GLEW_NV_fragment_program) GLEW_NV_fragment_program = !_glewInit_GL_NV_fragment_program(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_fragment_program */
#ifdef GL_NV_fragment_program2
  GLEW_NV_fragment_program2 = _glewSearchExtension("GL_NV_fragment_program2", extStart, extEnd);
#endif /* GL_NV_fragment_program2 */
#ifdef GL_NV_fragment_program4
  GLEW_NV_fragment_program4 = _glewSearchExtension("GL_NV_gpu_program4", extStart, extEnd);
#endif /* GL_NV_fragment_program4 */
#ifdef GL_NV_fragment_program_option
  GLEW_NV_fragment_program_option = _glewSearchExtension("GL_NV_fragment_program_option", extStart, extEnd);
#endif /* GL_NV_fragment_program_option */
#ifdef GL_NV_fragment_shader_interlock
  GLEW_NV_fragment_shader_interlock = _glewSearchExtension("GL_NV_fragment_shader_interlock", extStart, extEnd);
#endif /* GL_NV_fragment_shader_interlock */
#ifdef GL_NV_framebuffer_mixed_samples
  GLEW_NV_framebuffer_mixed_samples = _glewSearchExtension("GL_NV_framebuffer_mixed_samples", extStart, extEnd);
#endif /* GL_NV_framebuffer_mixed_samples */
#ifdef GL_NV_framebuffer_multisample_coverage
  GLEW_NV_framebuffer_multisample_coverage = _glewSearchExtension("GL_NV_framebuffer_multisample_coverage", extStart, extEnd);
  if (glewExperimental || GLEW_NV_framebuffer_multisample_coverage) GLEW_NV_framebuffer_multisample_coverage = !_glewInit_GL_NV_framebuffer_multisample_coverage(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_framebuffer_multisample_coverage */
#ifdef GL_NV_geometry_program4
  GLEW_NV_geometry_program4 = _glewSearchExtension("GL_NV_gpu_program4", extStart, extEnd);
  if (glewExperimental || GLEW_NV_geometry_program4) GLEW_NV_geometry_program4 = !_glewInit_GL_NV_geometry_program4(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_geometry_program4 */
#ifdef GL_NV_geometry_shader4
  GLEW_NV_geometry_shader4 = _glewSearchExtension("GL_NV_geometry_shader4", extStart, extEnd);
#endif /* GL_NV_geometry_shader4 */
#ifdef GL_NV_geometry_shader_passthrough
  GLEW_NV_geometry_shader_passthrough = _glewSearchExtension("GL_NV_geometry_shader_passthrough", extStart, extEnd);
#endif /* GL_NV_geometry_shader_passthrough */
#ifdef GL_NV_gpu_program4
  GLEW_NV_gpu_program4 = _glewSearchExtension("GL_NV_gpu_program4", extStart, extEnd);
  if (glewExperimental || GLEW_NV_gpu_program4) GLEW_NV_gpu_program4 = !_glewInit_GL_NV_gpu_program4(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_gpu_program4 */
#ifdef GL_NV_gpu_program5
  GLEW_NV_gpu_program5 = _glewSearchExtension("GL_NV_gpu_program5", extStart, extEnd);
#endif /* GL_NV_gpu_program5 */
#ifdef GL_NV_gpu_program5_mem_extended
  GLEW_NV_gpu_program5_mem_extended = _glewSearchExtension("GL_NV_gpu_program5_mem_extended", extStart, extEnd);
#endif /* GL_NV_gpu_program5_mem_extended */
#ifdef GL_NV_gpu_program_fp64
  GLEW_NV_gpu_program_fp64 = _glewSearchExtension("GL_NV_gpu_program_fp64", extStart, extEnd);
#endif /* GL_NV_gpu_program_fp64 */
#ifdef GL_NV_gpu_shader5
  GLEW_NV_gpu_shader5 = _glewSearchExtension("GL_NV_gpu_shader5", extStart, extEnd);
  if (glewExperimental || GLEW_NV_gpu_shader5) GLEW_NV_gpu_shader5 = !_glewInit_GL_NV_gpu_shader5(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_gpu_shader5 */
#ifdef GL_NV_half_float
  GLEW_NV_half_float = _glewSearchExtension("GL_NV_half_float", extStart, extEnd);
  if (glewExperimental || GLEW_NV_half_float) GLEW_NV_half_float = !_glewInit_GL_NV_half_float(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_half_float */
#ifdef GL_NV_internalformat_sample_query
  GLEW_NV_internalformat_sample_query = _glewSearchExtension("GL_NV_internalformat_sample_query", extStart, extEnd);
  if (glewExperimental || GLEW_NV_internalformat_sample_query) GLEW_NV_internalformat_sample_query = !_glewInit_GL_NV_internalformat_sample_query(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_internalformat_sample_query */
#ifdef GL_NV_light_max_exponent
  GLEW_NV_light_max_exponent = _glewSearchExtension("GL_NV_light_max_exponent", extStart, extEnd);
#endif /* GL_NV_light_max_exponent */
#ifdef GL_NV_multisample_coverage
  GLEW_NV_multisample_coverage = _glewSearchExtension("GL_NV_multisample_coverage", extStart, extEnd);
#endif /* GL_NV_multisample_coverage */
#ifdef GL_NV_multisample_filter_hint
  GLEW_NV_multisample_filter_hint = _glewSearchExtension("GL_NV_multisample_filter_hint", extStart, extEnd);
#endif /* GL_NV_multisample_filter_hint */
#ifdef GL_NV_occlusion_query
  GLEW_NV_occlusion_query = _glewSearchExtension("GL_NV_occlusion_query", extStart, extEnd);
  if (glewExperimental || GLEW_NV_occlusion_query) GLEW_NV_occlusion_query = !_glewInit_GL_NV_occlusion_query(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_occlusion_query */
#ifdef GL_NV_packed_depth_stencil
  GLEW_NV_packed_depth_stencil = _glewSearchExtension("GL_NV_packed_depth_stencil", extStart, extEnd);
#endif /* GL_NV_packed_depth_stencil */
#ifdef GL_NV_parameter_buffer_object
  GLEW_NV_parameter_buffer_object = _glewSearchExtension("GL_NV_parameter_buffer_object", extStart, extEnd);
  if (glewExperimental || GLEW_NV_parameter_buffer_object) GLEW_NV_parameter_buffer_object = !_glewInit_GL_NV_parameter_buffer_object(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_parameter_buffer_object */
#ifdef GL_NV_parameter_buffer_object2
  GLEW_NV_parameter_buffer_object2 = _glewSearchExtension("GL_NV_parameter_buffer_object2", extStart, extEnd);
#endif /* GL_NV_parameter_buffer_object2 */
#ifdef GL_NV_path_rendering
  GLEW_NV_path_rendering = _glewSearchExtension("GL_NV_path_rendering", extStart, extEnd);
  if (glewExperimental || GLEW_NV_path_rendering) GLEW_NV_path_rendering = !_glewInit_GL_NV_path_rendering(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_path_rendering */
#ifdef GL_NV_path_rendering_shared_edge
  GLEW_NV_path_rendering_shared_edge = _glewSearchExtension("GL_NV_path_rendering_shared_edge", extStart, extEnd);
#endif /* GL_NV_path_rendering_shared_edge */
#ifdef GL_NV_pixel_data_range
  GLEW_NV_pixel_data_range = _glewSearchExtension("GL_NV_pixel_data_range", extStart, extEnd);
  if (glewExperimental || GLEW_NV_pixel_data_range) GLEW_NV_pixel_data_range = !_glewInit_GL_NV_pixel_data_range(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_pixel_data_range */
#ifdef GL_NV_point_sprite
  GLEW_NV_point_sprite = _glewSearchExtension("GL_NV_point_sprite", extStart, extEnd);
  if (glewExperimental || GLEW_NV_point_sprite) GLEW_NV_point_sprite = !_glewInit_GL_NV_point_sprite(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_point_sprite */
#ifdef GL_NV_present_video
  GLEW_NV_present_video = _glewSearchExtension("GL_NV_present_video", extStart, extEnd);
  if (glewExperimental || GLEW_NV_present_video) GLEW_NV_present_video = !_glewInit_GL_NV_present_video(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_present_video */
#ifdef GL_NV_primitive_restart
  GLEW_NV_primitive_restart = _glewSearchExtension("GL_NV_primitive_restart", extStart, extEnd);
  if (glewExperimental || GLEW_NV_primitive_restart) GLEW_NV_primitive_restart = !_glewInit_GL_NV_primitive_restart(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_primitive_restart */
#ifdef GL_NV_register_combiners
  GLEW_NV_register_combiners = _glewSearchExtension("GL_NV_register_combiners", extStart, extEnd);
  if (glewExperimental || GLEW_NV_register_combiners) GLEW_NV_register_combiners = !_glewInit_GL_NV_register_combiners(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_register_combiners */
#ifdef GL_NV_register_combiners2
  GLEW_NV_register_combiners2 = _glewSearchExtension("GL_NV_register_combiners2", extStart, extEnd);
  if (glewExperimental || GLEW_NV_register_combiners2) GLEW_NV_register_combiners2 = !_glewInit_GL_NV_register_combiners2(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_register_combiners2 */
#ifdef GL_NV_sample_locations
  GLEW_NV_sample_locations = _glewSearchExtension("GL_NV_sample_locations", extStart, extEnd);
  if (glewExperimental || GLEW_NV_sample_locations) GLEW_NV_sample_locations = !_glewInit_GL_NV_sample_locations(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_sample_locations */
#ifdef GL_NV_sample_mask_override_coverage
  GLEW_NV_sample_mask_override_coverage = _glewSearchExtension("GL_NV_sample_mask_override_coverage", extStart, extEnd);
#endif /* GL_NV_sample_mask_override_coverage */
#ifdef GL_NV_shader_atomic_counters
  GLEW_NV_shader_atomic_counters = _glewSearchExtension("GL_NV_shader_atomic_counters", extStart, extEnd);
#endif /* GL_NV_shader_atomic_counters */
#ifdef GL_NV_shader_atomic_float
  GLEW_NV_shader_atomic_float = _glewSearchExtension("GL_NV_shader_atomic_float", extStart, extEnd);
#endif /* GL_NV_shader_atomic_float */
#ifdef GL_NV_shader_atomic_fp16_vector
  GLEW_NV_shader_atomic_fp16_vector = _glewSearchExtension("GL_NV_shader_atomic_fp16_vector", extStart, extEnd);
#endif /* GL_NV_shader_atomic_fp16_vector */
#ifdef GL_NV_shader_atomic_int64
  GLEW_NV_shader_atomic_int64 = _glewSearchExtension("GL_NV_shader_atomic_int64", extStart, extEnd);
#endif /* GL_NV_shader_atomic_int64 */
#ifdef GL_NV_shader_buffer_load
  GLEW_NV_shader_buffer_load = _glewSearchExtension("GL_NV_shader_buffer_load", extStart, extEnd);
  if (glewExperimental || GLEW_NV_shader_buffer_load) GLEW_NV_shader_buffer_load = !_glewInit_GL_NV_shader_buffer_load(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_shader_buffer_load */
#ifdef GL_NV_shader_storage_buffer_object
  GLEW_NV_shader_storage_buffer_object = _glewSearchExtension("GL_NV_shader_storage_buffer_object", extStart, extEnd);
#endif /* GL_NV_shader_storage_buffer_object */
#ifdef GL_NV_shader_thread_group
  GLEW_NV_shader_thread_group = _glewSearchExtension("GL_NV_shader_thread_group", extStart, extEnd);
#endif /* GL_NV_shader_thread_group */
#ifdef GL_NV_shader_thread_shuffle
  GLEW_NV_shader_thread_shuffle = _glewSearchExtension("GL_NV_shader_thread_shuffle", extStart, extEnd);
#endif /* GL_NV_shader_thread_shuffle */
#ifdef GL_NV_tessellation_program5
  GLEW_NV_tessellation_program5 = _glewSearchExtension("GL_NV_gpu_program5", extStart, extEnd);
#endif /* GL_NV_tessellation_program5 */
#ifdef GL_NV_texgen_emboss
  GLEW_NV_texgen_emboss = _glewSearchExtension("GL_NV_texgen_emboss", extStart, extEnd);
#endif /* GL_NV_texgen_emboss */
#ifdef GL_NV_texgen_reflection
  GLEW_NV_texgen_reflection = _glewSearchExtension("GL_NV_texgen_reflection", extStart, extEnd);
#endif /* GL_NV_texgen_reflection */
#ifdef GL_NV_texture_barrier
  GLEW_NV_texture_barrier = _glewSearchExtension("GL_NV_texture_barrier", extStart, extEnd);
  if (glewExperimental || GLEW_NV_texture_barrier) GLEW_NV_texture_barrier = !_glewInit_GL_NV_texture_barrier(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_texture_barrier */
#ifdef GL_NV_texture_compression_vtc
  GLEW_NV_texture_compression_vtc = _glewSearchExtension("GL_NV_texture_compression_vtc", extStart, extEnd);
#endif /* GL_NV_texture_compression_vtc */
#ifdef GL_NV_texture_env_combine4
  GLEW_NV_texture_env_combine4 = _glewSearchExtension("GL_NV_texture_env_combine4", extStart, extEnd);
#endif /* GL_NV_texture_env_combine4 */
#ifdef GL_NV_texture_expand_normal
  GLEW_NV_texture_expand_normal = _glewSearchExtension("GL_NV_texture_expand_normal", extStart, extEnd);
#endif /* GL_NV_texture_expand_normal */
#ifdef GL_NV_texture_multisample
  GLEW_NV_texture_multisample = _glewSearchExtension("GL_NV_texture_multisample", extStart, extEnd);
  if (glewExperimental || GLEW_NV_texture_multisample) GLEW_NV_texture_multisample = !_glewInit_GL_NV_texture_multisample(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_texture_multisample */
#ifdef GL_NV_texture_rectangle
  GLEW_NV_texture_rectangle = _glewSearchExtension("GL_NV_texture_rectangle", extStart, extEnd);
#endif /* GL_NV_texture_rectangle */
#ifdef GL_NV_texture_shader
  GLEW_NV_texture_shader = _glewSearchExtension("GL_NV_texture_shader", extStart, extEnd);
#endif /* GL_NV_texture_shader */
#ifdef GL_NV_texture_shader2
  GLEW_NV_texture_shader2 = _glewSearchExtension("GL_NV_texture_shader2", extStart, extEnd);
#endif /* GL_NV_texture_shader2 */
#ifdef GL_NV_texture_shader3
  GLEW_NV_texture_shader3 = _glewSearchExtension("GL_NV_texture_shader3", extStart, extEnd);
#endif /* GL_NV_texture_shader3 */
#ifdef GL_NV_transform_feedback
  GLEW_NV_transform_feedback = _glewSearchExtension("GL_NV_transform_feedback", extStart, extEnd);
  if (glewExperimental || GLEW_NV_transform_feedback) GLEW_NV_transform_feedback = !_glewInit_GL_NV_transform_feedback(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_transform_feedback */
#ifdef GL_NV_transform_feedback2
  GLEW_NV_transform_feedback2 = _glewSearchExtension("GL_NV_transform_feedback2", extStart, extEnd);
  if (glewExperimental || GLEW_NV_transform_feedback2) GLEW_NV_transform_feedback2 = !_glewInit_GL_NV_transform_feedback2(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_transform_feedback2 */
#ifdef GL_NV_uniform_buffer_unified_memory
  GLEW_NV_uniform_buffer_unified_memory = _glewSearchExtension("GL_NV_uniform_buffer_unified_memory", extStart, extEnd);
#endif /* GL_NV_uniform_buffer_unified_memory */
#ifdef GL_NV_vdpau_interop
  GLEW_NV_vdpau_interop = _glewSearchExtension("GL_NV_vdpau_interop", extStart, extEnd);
  if (glewExperimental || GLEW_NV_vdpau_interop) GLEW_NV_vdpau_interop = !_glewInit_GL_NV_vdpau_interop(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_vdpau_interop */
#ifdef GL_NV_vertex_array_range
  GLEW_NV_vertex_array_range = _glewSearchExtension("GL_NV_vertex_array_range", extStart, extEnd);
  if (glewExperimental || GLEW_NV_vertex_array_range) GLEW_NV_vertex_array_range = !_glewInit_GL_NV_vertex_array_range(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_vertex_array_range */
#ifdef GL_NV_vertex_array_range2
  GLEW_NV_vertex_array_range2 = _glewSearchExtension("GL_NV_vertex_array_range2", extStart, extEnd);
#endif /* GL_NV_vertex_array_range2 */
#ifdef GL_NV_vertex_attrib_integer_64bit
  GLEW_NV_vertex_attrib_integer_64bit = _glewSearchExtension("GL_NV_vertex_attrib_integer_64bit", extStart, extEnd);
  if (glewExperimental || GLEW_NV_vertex_attrib_integer_64bit) GLEW_NV_vertex_attrib_integer_64bit = !_glewInit_GL_NV_vertex_attrib_integer_64bit(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_vertex_attrib_integer_64bit */
#ifdef GL_NV_vertex_buffer_unified_memory
  GLEW_NV_vertex_buffer_unified_memory = _glewSearchExtension("GL_NV_vertex_buffer_unified_memory", extStart, extEnd);
  if (glewExperimental || GLEW_NV_vertex_buffer_unified_memory) GLEW_NV_vertex_buffer_unified_memory = !_glewInit_GL_NV_vertex_buffer_unified_memory(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_vertex_buffer_unified_memory */
#ifdef GL_NV_vertex_program
  GLEW_NV_vertex_program = _glewSearchExtension("GL_NV_vertex_program", extStart, extEnd);
  if (glewExperimental || GLEW_NV_vertex_program) GLEW_NV_vertex_program = !_glewInit_GL_NV_vertex_program(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_vertex_program */
#ifdef GL_NV_vertex_program1_1
  GLEW_NV_vertex_program1_1 = _glewSearchExtension("GL_NV_vertex_program1_1", extStart, extEnd);
#endif /* GL_NV_vertex_program1_1 */
#ifdef GL_NV_vertex_program2
  GLEW_NV_vertex_program2 = _glewSearchExtension("GL_NV_vertex_program2", extStart, extEnd);
#endif /* GL_NV_vertex_program2 */
#ifdef GL_NV_vertex_program2_option
  GLEW_NV_vertex_program2_option = _glewSearchExtension("GL_NV_vertex_program2_option", extStart, extEnd);
#endif /* GL_NV_vertex_program2_option */
#ifdef GL_NV_vertex_program3
  GLEW_NV_vertex_program3 = _glewSearchExtension("GL_NV_vertex_program3", extStart, extEnd);
#endif /* GL_NV_vertex_program3 */
#ifdef GL_NV_vertex_program4
  GLEW_NV_vertex_program4 = _glewSearchExtension("GL_NV_gpu_program4", extStart, extEnd);
#endif /* GL_NV_vertex_program4 */
#ifdef GL_NV_video_capture
  GLEW_NV_video_capture = _glewSearchExtension("GL_NV_video_capture", extStart, extEnd);
  if (glewExperimental || GLEW_NV_video_capture) GLEW_NV_video_capture = !_glewInit_GL_NV_video_capture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_NV_video_capture */
#ifdef GL_NV_viewport_array2
  GLEW_NV_viewport_array2 = _glewSearchExtension("GL_NV_viewport_array2", extStart, extEnd);
#endif /* GL_NV_viewport_array2 */
#ifdef GL_OES_byte_coordinates
  GLEW_OES_byte_coordinates = _glewSearchExtension("GL_OES_byte_coordinates", extStart, extEnd);
#endif /* GL_OES_byte_coordinates */
#ifdef GL_OES_compressed_paletted_texture
  GLEW_OES_compressed_paletted_texture = _glewSearchExtension("GL_OES_compressed_paletted_texture", extStart, extEnd);
#endif /* GL_OES_compressed_paletted_texture */
#ifdef GL_OES_read_format
  GLEW_OES_read_format = _glewSearchExtension("GL_OES_read_format", extStart, extEnd);
#endif /* GL_OES_read_format */
#ifdef GL_OES_single_precision
  GLEW_OES_single_precision = _glewSearchExtension("GL_OES_single_precision", extStart, extEnd);
  if (glewExperimental || GLEW_OES_single_precision) GLEW_OES_single_precision = !_glewInit_GL_OES_single_precision(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_OES_single_precision */
#ifdef GL_OML_interlace
  GLEW_OML_interlace = _glewSearchExtension("GL_OML_interlace", extStart, extEnd);
#endif /* GL_OML_interlace */
#ifdef GL_OML_resample
  GLEW_OML_resample = _glewSearchExtension("GL_OML_resample", extStart, extEnd);
#endif /* GL_OML_resample */
#ifdef GL_OML_subsample
  GLEW_OML_subsample = _glewSearchExtension("GL_OML_subsample", extStart, extEnd);
#endif /* GL_OML_subsample */
#ifdef GL_OVR_multiview
  GLEW_OVR_multiview = _glewSearchExtension("GL_OVR_multiview", extStart, extEnd);
  if (glewExperimental || GLEW_OVR_multiview) GLEW_OVR_multiview = !_glewInit_GL_OVR_multiview(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_OVR_multiview */
#ifdef GL_OVR_multiview2
  GLEW_OVR_multiview2 = _glewSearchExtension("GL_OVR_multiview2", extStart, extEnd);
#endif /* GL_OVR_multiview2 */
#ifdef GL_PGI_misc_hints
  GLEW_PGI_misc_hints = _glewSearchExtension("GL_PGI_misc_hints", extStart, extEnd);
#endif /* GL_PGI_misc_hints */
#ifdef GL_PGI_vertex_hints
  GLEW_PGI_vertex_hints = _glewSearchExtension("GL_PGI_vertex_hints", extStart, extEnd);
#endif /* GL_PGI_vertex_hints */
#ifdef GL_REGAL_ES1_0_compatibility
  GLEW_REGAL_ES1_0_compatibility = _glewSearchExtension("GL_REGAL_ES1_0_compatibility", extStart, extEnd);
  if (glewExperimental || GLEW_REGAL_ES1_0_compatibility) GLEW_REGAL_ES1_0_compatibility = !_glewInit_GL_REGAL_ES1_0_compatibility(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_REGAL_ES1_0_compatibility */
#ifdef GL_REGAL_ES1_1_compatibility
  GLEW_REGAL_ES1_1_compatibility = _glewSearchExtension("GL_REGAL_ES1_1_compatibility", extStart, extEnd);
  if (glewExperimental || GLEW_REGAL_ES1_1_compatibility) GLEW_REGAL_ES1_1_compatibility = !_glewInit_GL_REGAL_ES1_1_compatibility(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_REGAL_ES1_1_compatibility */
#ifdef GL_REGAL_enable
  GLEW_REGAL_enable = _glewSearchExtension("GL_REGAL_enable", extStart, extEnd);
#endif /* GL_REGAL_enable */
#ifdef GL_REGAL_error_string
  GLEW_REGAL_error_string = _glewSearchExtension("GL_REGAL_error_string", extStart, extEnd);
  if (glewExperimental || GLEW_REGAL_error_string) GLEW_REGAL_error_string = !_glewInit_GL_REGAL_error_string(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_REGAL_error_string */
#ifdef GL_REGAL_extension_query
  GLEW_REGAL_extension_query = _glewSearchExtension("GL_REGAL_extension_query", extStart, extEnd);
  if (glewExperimental || GLEW_REGAL_extension_query) GLEW_REGAL_extension_query = !_glewInit_GL_REGAL_extension_query(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_REGAL_extension_query */
#ifdef GL_REGAL_log
  GLEW_REGAL_log = _glewSearchExtension("GL_REGAL_log", extStart, extEnd);
  if (glewExperimental || GLEW_REGAL_log) GLEW_REGAL_log = !_glewInit_GL_REGAL_log(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_REGAL_log */
#ifdef GL_REGAL_proc_address
  GLEW_REGAL_proc_address = _glewSearchExtension("GL_REGAL_proc_address", extStart, extEnd);
  if (glewExperimental || GLEW_REGAL_proc_address) GLEW_REGAL_proc_address = !_glewInit_GL_REGAL_proc_address(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_REGAL_proc_address */
#ifdef GL_REND_screen_coordinates
  GLEW_REND_screen_coordinates = _glewSearchExtension("GL_REND_screen_coordinates", extStart, extEnd);
#endif /* GL_REND_screen_coordinates */
#ifdef GL_S3_s3tc
  GLEW_S3_s3tc = _glewSearchExtension("GL_S3_s3tc", extStart, extEnd);
#endif /* GL_S3_s3tc */
#ifdef GL_SGIS_color_range
  GLEW_SGIS_color_range = _glewSearchExtension("GL_SGIS_color_range", extStart, extEnd);
#endif /* GL_SGIS_color_range */
#ifdef GL_SGIS_detail_texture
  GLEW_SGIS_detail_texture = _glewSearchExtension("GL_SGIS_detail_texture", extStart, extEnd);
  if (glewExperimental || GLEW_SGIS_detail_texture) GLEW_SGIS_detail_texture = !_glewInit_GL_SGIS_detail_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIS_detail_texture */
#ifdef GL_SGIS_fog_function
  GLEW_SGIS_fog_function = _glewSearchExtension("GL_SGIS_fog_function", extStart, extEnd);
  if (glewExperimental || GLEW_SGIS_fog_function) GLEW_SGIS_fog_function = !_glewInit_GL_SGIS_fog_function(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIS_fog_function */
#ifdef GL_SGIS_generate_mipmap
  GLEW_SGIS_generate_mipmap = _glewSearchExtension("GL_SGIS_generate_mipmap", extStart, extEnd);
#endif /* GL_SGIS_generate_mipmap */
#ifdef GL_SGIS_multisample
  GLEW_SGIS_multisample = _glewSearchExtension("GL_SGIS_multisample", extStart, extEnd);
  if (glewExperimental || GLEW_SGIS_multisample) GLEW_SGIS_multisample = !_glewInit_GL_SGIS_multisample(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIS_multisample */
#ifdef GL_SGIS_pixel_texture
  GLEW_SGIS_pixel_texture = _glewSearchExtension("GL_SGIS_pixel_texture", extStart, extEnd);
#endif /* GL_SGIS_pixel_texture */
#ifdef GL_SGIS_point_line_texgen
  GLEW_SGIS_point_line_texgen = _glewSearchExtension("GL_SGIS_point_line_texgen", extStart, extEnd);
#endif /* GL_SGIS_point_line_texgen */
#ifdef GL_SGIS_sharpen_texture
  GLEW_SGIS_sharpen_texture = _glewSearchExtension("GL_SGIS_sharpen_texture", extStart, extEnd);
  if (glewExperimental || GLEW_SGIS_sharpen_texture) GLEW_SGIS_sharpen_texture = !_glewInit_GL_SGIS_sharpen_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIS_sharpen_texture */
#ifdef GL_SGIS_texture4D
  GLEW_SGIS_texture4D = _glewSearchExtension("GL_SGIS_texture4D", extStart, extEnd);
  if (glewExperimental || GLEW_SGIS_texture4D) GLEW_SGIS_texture4D = !_glewInit_GL_SGIS_texture4D(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIS_texture4D */
#ifdef GL_SGIS_texture_border_clamp
  GLEW_SGIS_texture_border_clamp = _glewSearchExtension("GL_SGIS_texture_border_clamp", extStart, extEnd);
#endif /* GL_SGIS_texture_border_clamp */
#ifdef GL_SGIS_texture_edge_clamp
  GLEW_SGIS_texture_edge_clamp = _glewSearchExtension("GL_SGIS_texture_edge_clamp", extStart, extEnd);
#endif /* GL_SGIS_texture_edge_clamp */
#ifdef GL_SGIS_texture_filter4
  GLEW_SGIS_texture_filter4 = _glewSearchExtension("GL_SGIS_texture_filter4", extStart, extEnd);
  if (glewExperimental || GLEW_SGIS_texture_filter4) GLEW_SGIS_texture_filter4 = !_glewInit_GL_SGIS_texture_filter4(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIS_texture_filter4 */
#ifdef GL_SGIS_texture_lod
  GLEW_SGIS_texture_lod = _glewSearchExtension("GL_SGIS_texture_lod", extStart, extEnd);
#endif /* GL_SGIS_texture_lod */
#ifdef GL_SGIS_texture_select
  GLEW_SGIS_texture_select = _glewSearchExtension("GL_SGIS_texture_select", extStart, extEnd);
#endif /* GL_SGIS_texture_select */
#ifdef GL_SGIX_async
  GLEW_SGIX_async = _glewSearchExtension("GL_SGIX_async", extStart, extEnd);
  if (glewExperimental || GLEW_SGIX_async) GLEW_SGIX_async = !_glewInit_GL_SGIX_async(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIX_async */
#ifdef GL_SGIX_async_histogram
  GLEW_SGIX_async_histogram = _glewSearchExtension("GL_SGIX_async_histogram", extStart, extEnd);
#endif /* GL_SGIX_async_histogram */
#ifdef GL_SGIX_async_pixel
  GLEW_SGIX_async_pixel = _glewSearchExtension("GL_SGIX_async_pixel", extStart, extEnd);
#endif /* GL_SGIX_async_pixel */
#ifdef GL_SGIX_blend_alpha_minmax
  GLEW_SGIX_blend_alpha_minmax = _glewSearchExtension("GL_SGIX_blend_alpha_minmax", extStart, extEnd);
#endif /* GL_SGIX_blend_alpha_minmax */
#ifdef GL_SGIX_clipmap
  GLEW_SGIX_clipmap = _glewSearchExtension("GL_SGIX_clipmap", extStart, extEnd);
#endif /* GL_SGIX_clipmap */
#ifdef GL_SGIX_convolution_accuracy
  GLEW_SGIX_convolution_accuracy = _glewSearchExtension("GL_SGIX_convolution_accuracy", extStart, extEnd);
#endif /* GL_SGIX_convolution_accuracy */
#ifdef GL_SGIX_depth_texture
  GLEW_SGIX_depth_texture = _glewSearchExtension("GL_SGIX_depth_texture", extStart, extEnd);
#endif /* GL_SGIX_depth_texture */
#ifdef GL_SGIX_flush_raster
  GLEW_SGIX_flush_raster = _glewSearchExtension("GL_SGIX_flush_raster", extStart, extEnd);
  if (glewExperimental || GLEW_SGIX_flush_raster) GLEW_SGIX_flush_raster = !_glewInit_GL_SGIX_flush_raster(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIX_flush_raster */
#ifdef GL_SGIX_fog_offset
  GLEW_SGIX_fog_offset = _glewSearchExtension("GL_SGIX_fog_offset", extStart, extEnd);
#endif /* GL_SGIX_fog_offset */
#ifdef GL_SGIX_fog_texture
  GLEW_SGIX_fog_texture = _glewSearchExtension("GL_SGIX_fog_texture", extStart, extEnd);
  if (glewExperimental || GLEW_SGIX_fog_texture) GLEW_SGIX_fog_texture = !_glewInit_GL_SGIX_fog_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIX_fog_texture */
#ifdef GL_SGIX_fragment_specular_lighting
  GLEW_SGIX_fragment_specular_lighting = _glewSearchExtension("GL_SGIX_fragment_specular_lighting", extStart, extEnd);
  if (glewExperimental || GLEW_SGIX_fragment_specular_lighting) GLEW_SGIX_fragment_specular_lighting = !_glewInit_GL_SGIX_fragment_specular_lighting(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIX_fragment_specular_lighting */
#ifdef GL_SGIX_framezoom
  GLEW_SGIX_framezoom = _glewSearchExtension("GL_SGIX_framezoom", extStart, extEnd);
  if (glewExperimental || GLEW_SGIX_framezoom) GLEW_SGIX_framezoom = !_glewInit_GL_SGIX_framezoom(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIX_framezoom */
#ifdef GL_SGIX_interlace
  GLEW_SGIX_interlace = _glewSearchExtension("GL_SGIX_interlace", extStart, extEnd);
#endif /* GL_SGIX_interlace */
#ifdef GL_SGIX_ir_instrument1
  GLEW_SGIX_ir_instrument1 = _glewSearchExtension("GL_SGIX_ir_instrument1", extStart, extEnd);
#endif /* GL_SGIX_ir_instrument1 */
#ifdef GL_SGIX_list_priority
  GLEW_SGIX_list_priority = _glewSearchExtension("GL_SGIX_list_priority", extStart, extEnd);
#endif /* GL_SGIX_list_priority */
#ifdef GL_SGIX_pixel_texture
  GLEW_SGIX_pixel_texture = _glewSearchExtension("GL_SGIX_pixel_texture", extStart, extEnd);
  if (glewExperimental || GLEW_SGIX_pixel_texture) GLEW_SGIX_pixel_texture = !_glewInit_GL_SGIX_pixel_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIX_pixel_texture */
#ifdef GL_SGIX_pixel_texture_bits
  GLEW_SGIX_pixel_texture_bits = _glewSearchExtension("GL_SGIX_pixel_texture_bits", extStart, extEnd);
#endif /* GL_SGIX_pixel_texture_bits */
#ifdef GL_SGIX_reference_plane
  GLEW_SGIX_reference_plane = _glewSearchExtension("GL_SGIX_reference_plane", extStart, extEnd);
  if (glewExperimental || GLEW_SGIX_reference_plane) GLEW_SGIX_reference_plane = !_glewInit_GL_SGIX_reference_plane(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIX_reference_plane */
#ifdef GL_SGIX_resample
  GLEW_SGIX_resample = _glewSearchExtension("GL_SGIX_resample", extStart, extEnd);
#endif /* GL_SGIX_resample */
#ifdef GL_SGIX_shadow
  GLEW_SGIX_shadow = _glewSearchExtension("GL_SGIX_shadow", extStart, extEnd);
#endif /* GL_SGIX_shadow */
#ifdef GL_SGIX_shadow_ambient
  GLEW_SGIX_shadow_ambient = _glewSearchExtension("GL_SGIX_shadow_ambient", extStart, extEnd);
#endif /* GL_SGIX_shadow_ambient */
#ifdef GL_SGIX_sprite
  GLEW_SGIX_sprite = _glewSearchExtension("GL_SGIX_sprite", extStart, extEnd);
  if (glewExperimental || GLEW_SGIX_sprite) GLEW_SGIX_sprite = !_glewInit_GL_SGIX_sprite(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIX_sprite */
#ifdef GL_SGIX_tag_sample_buffer
  GLEW_SGIX_tag_sample_buffer = _glewSearchExtension("GL_SGIX_tag_sample_buffer", extStart, extEnd);
  if (glewExperimental || GLEW_SGIX_tag_sample_buffer) GLEW_SGIX_tag_sample_buffer = !_glewInit_GL_SGIX_tag_sample_buffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGIX_tag_sample_buffer */
#ifdef GL_SGIX_texture_add_env
  GLEW_SGIX_texture_add_env = _glewSearchExtension("GL_SGIX_texture_add_env", extStart, extEnd);
#endif /* GL_SGIX_texture_add_env */
#ifdef GL_SGIX_texture_coordinate_clamp
  GLEW_SGIX_texture_coordinate_clamp = _glewSearchExtension("GL_SGIX_texture_coordinate_clamp", extStart, extEnd);
#endif /* GL_SGIX_texture_coordinate_clamp */
#ifdef GL_SGIX_texture_lod_bias
  GLEW_SGIX_texture_lod_bias = _glewSearchExtension("GL_SGIX_texture_lod_bias", extStart, extEnd);
#endif /* GL_SGIX_texture_lod_bias */
#ifdef GL_SGIX_texture_multi_buffer
  GLEW_SGIX_texture_multi_buffer = _glewSearchExtension("GL_SGIX_texture_multi_buffer", extStart, extEnd);
#endif /* GL_SGIX_texture_multi_buffer */
#ifdef GL_SGIX_texture_range
  GLEW_SGIX_texture_range = _glewSearchExtension("GL_SGIX_texture_range", extStart, extEnd);
#endif /* GL_SGIX_texture_range */
#ifdef GL_SGIX_texture_scale_bias
  GLEW_SGIX_texture_scale_bias = _glewSearchExtension("GL_SGIX_texture_scale_bias", extStart, extEnd);
#endif /* GL_SGIX_texture_scale_bias */
#ifdef GL_SGIX_vertex_preclip
  GLEW_SGIX_vertex_preclip = _glewSearchExtension("GL_SGIX_vertex_preclip", extStart, extEnd);
#endif /* GL_SGIX_vertex_preclip */
#ifdef GL_SGIX_vertex_preclip_hint
  GLEW_SGIX_vertex_preclip_hint = _glewSearchExtension("GL_SGIX_vertex_preclip_hint", extStart, extEnd);
#endif /* GL_SGIX_vertex_preclip_hint */
#ifdef GL_SGIX_ycrcb
  GLEW_SGIX_ycrcb = _glewSearchExtension("GL_SGIX_ycrcb", extStart, extEnd);
#endif /* GL_SGIX_ycrcb */
#ifdef GL_SGI_color_matrix
  GLEW_SGI_color_matrix = _glewSearchExtension("GL_SGI_color_matrix", extStart, extEnd);
#endif /* GL_SGI_color_matrix */
#ifdef GL_SGI_color_table
  GLEW_SGI_color_table = _glewSearchExtension("GL_SGI_color_table", extStart, extEnd);
  if (glewExperimental || GLEW_SGI_color_table) GLEW_SGI_color_table = !_glewInit_GL_SGI_color_table(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SGI_color_table */
#ifdef GL_SGI_texture_color_table
  GLEW_SGI_texture_color_table = _glewSearchExtension("GL_SGI_texture_color_table", extStart, extEnd);
#endif /* GL_SGI_texture_color_table */
#ifdef GL_SUNX_constant_data
  GLEW_SUNX_constant_data = _glewSearchExtension("GL_SUNX_constant_data", extStart, extEnd);
  if (glewExperimental || GLEW_SUNX_constant_data) GLEW_SUNX_constant_data = !_glewInit_GL_SUNX_constant_data(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SUNX_constant_data */
#ifdef GL_SUN_convolution_border_modes
  GLEW_SUN_convolution_border_modes = _glewSearchExtension("GL_SUN_convolution_border_modes", extStart, extEnd);
#endif /* GL_SUN_convolution_border_modes */
#ifdef GL_SUN_global_alpha
  GLEW_SUN_global_alpha = _glewSearchExtension("GL_SUN_global_alpha", extStart, extEnd);
  if (glewExperimental || GLEW_SUN_global_alpha) GLEW_SUN_global_alpha = !_glewInit_GL_SUN_global_alpha(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SUN_global_alpha */
#ifdef GL_SUN_mesh_array
  GLEW_SUN_mesh_array = _glewSearchExtension("GL_SUN_mesh_array", extStart, extEnd);
#endif /* GL_SUN_mesh_array */
#ifdef GL_SUN_read_video_pixels
  GLEW_SUN_read_video_pixels = _glewSearchExtension("GL_SUN_read_video_pixels", extStart, extEnd);
  if (glewExperimental || GLEW_SUN_read_video_pixels) GLEW_SUN_read_video_pixels = !_glewInit_GL_SUN_read_video_pixels(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SUN_read_video_pixels */
#ifdef GL_SUN_slice_accum
  GLEW_SUN_slice_accum = _glewSearchExtension("GL_SUN_slice_accum", extStart, extEnd);
#endif /* GL_SUN_slice_accum */
#ifdef GL_SUN_triangle_list
  GLEW_SUN_triangle_list = _glewSearchExtension("GL_SUN_triangle_list", extStart, extEnd);
  if (glewExperimental || GLEW_SUN_triangle_list) GLEW_SUN_triangle_list = !_glewInit_GL_SUN_triangle_list(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SUN_triangle_list */
#ifdef GL_SUN_vertex
  GLEW_SUN_vertex = _glewSearchExtension("GL_SUN_vertex", extStart, extEnd);
  if (glewExperimental || GLEW_SUN_vertex) GLEW_SUN_vertex = !_glewInit_GL_SUN_vertex(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_SUN_vertex */
#ifdef GL_WIN_phong_shading
  GLEW_WIN_phong_shading = _glewSearchExtension("GL_WIN_phong_shading", extStart, extEnd);
#endif /* GL_WIN_phong_shading */
#ifdef GL_WIN_specular_fog
  GLEW_WIN_specular_fog = _glewSearchExtension("GL_WIN_specular_fog", extStart, extEnd);
#endif /* GL_WIN_specular_fog */
#ifdef GL_WIN_swap_hint
  GLEW_WIN_swap_hint = _glewSearchExtension("GL_WIN_swap_hint", extStart, extEnd);
  if (glewExperimental || GLEW_WIN_swap_hint) GLEW_WIN_swap_hint = !_glewInit_GL_WIN_swap_hint(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GL_WIN_swap_hint */

  return GLEW_OK;
}


#if defined(_WIN32)

#if !defined(GLEW_MX)

PFNWGLSETSTEREOEMITTERSTATE3DLPROC __wglewSetStereoEmitterState3DL = NULL;

PFNWGLBLITCONTEXTFRAMEBUFFERAMDPROC __wglewBlitContextFramebufferAMD = NULL;
PFNWGLCREATEASSOCIATEDCONTEXTAMDPROC __wglewCreateAssociatedContextAMD = NULL;
PFNWGLCREATEASSOCIATEDCONTEXTATTRIBSAMDPROC __wglewCreateAssociatedContextAttribsAMD = NULL;
PFNWGLDELETEASSOCIATEDCONTEXTAMDPROC __wglewDeleteAssociatedContextAMD = NULL;
PFNWGLGETCONTEXTGPUIDAMDPROC __wglewGetContextGPUIDAMD = NULL;
PFNWGLGETCURRENTASSOCIATEDCONTEXTAMDPROC __wglewGetCurrentAssociatedContextAMD = NULL;
PFNWGLGETGPUIDSAMDPROC __wglewGetGPUIDsAMD = NULL;
PFNWGLGETGPUINFOAMDPROC __wglewGetGPUInfoAMD = NULL;
PFNWGLMAKEASSOCIATEDCONTEXTCURRENTAMDPROC __wglewMakeAssociatedContextCurrentAMD = NULL;

PFNWGLCREATEBUFFERREGIONARBPROC __wglewCreateBufferRegionARB = NULL;
PFNWGLDELETEBUFFERREGIONARBPROC __wglewDeleteBufferRegionARB = NULL;
PFNWGLRESTOREBUFFERREGIONARBPROC __wglewRestoreBufferRegionARB = NULL;
PFNWGLSAVEBUFFERREGIONARBPROC __wglewSaveBufferRegionARB = NULL;

PFNWGLCREATECONTEXTATTRIBSARBPROC __wglewCreateContextAttribsARB = NULL;

PFNWGLGETEXTENSIONSSTRINGARBPROC __wglewGetExtensionsStringARB = NULL;

PFNWGLGETCURRENTREADDCARBPROC __wglewGetCurrentReadDCARB = NULL;
PFNWGLMAKECONTEXTCURRENTARBPROC __wglewMakeContextCurrentARB = NULL;

PFNWGLCREATEPBUFFERARBPROC __wglewCreatePbufferARB = NULL;
PFNWGLDESTROYPBUFFERARBPROC __wglewDestroyPbufferARB = NULL;
PFNWGLGETPBUFFERDCARBPROC __wglewGetPbufferDCARB = NULL;
PFNWGLQUERYPBUFFERARBPROC __wglewQueryPbufferARB = NULL;
PFNWGLRELEASEPBUFFERDCARBPROC __wglewReleasePbufferDCARB = NULL;

PFNWGLCHOOSEPIXELFORMATARBPROC __wglewChoosePixelFormatARB = NULL;
PFNWGLGETPIXELFORMATATTRIBFVARBPROC __wglewGetPixelFormatAttribfvARB = NULL;
PFNWGLGETPIXELFORMATATTRIBIVARBPROC __wglewGetPixelFormatAttribivARB = NULL;

PFNWGLBINDTEXIMAGEARBPROC __wglewBindTexImageARB = NULL;
PFNWGLRELEASETEXIMAGEARBPROC __wglewReleaseTexImageARB = NULL;
PFNWGLSETPBUFFERATTRIBARBPROC __wglewSetPbufferAttribARB = NULL;

PFNWGLBINDDISPLAYCOLORTABLEEXTPROC __wglewBindDisplayColorTableEXT = NULL;
PFNWGLCREATEDISPLAYCOLORTABLEEXTPROC __wglewCreateDisplayColorTableEXT = NULL;
PFNWGLDESTROYDISPLAYCOLORTABLEEXTPROC __wglewDestroyDisplayColorTableEXT = NULL;
PFNWGLLOADDISPLAYCOLORTABLEEXTPROC __wglewLoadDisplayColorTableEXT = NULL;

PFNWGLGETEXTENSIONSSTRINGEXTPROC __wglewGetExtensionsStringEXT = NULL;

PFNWGLGETCURRENTREADDCEXTPROC __wglewGetCurrentReadDCEXT = NULL;
PFNWGLMAKECONTEXTCURRENTEXTPROC __wglewMakeContextCurrentEXT = NULL;

PFNWGLCREATEPBUFFEREXTPROC __wglewCreatePbufferEXT = NULL;
PFNWGLDESTROYPBUFFEREXTPROC __wglewDestroyPbufferEXT = NULL;
PFNWGLGETPBUFFERDCEXTPROC __wglewGetPbufferDCEXT = NULL;
PFNWGLQUERYPBUFFEREXTPROC __wglewQueryPbufferEXT = NULL;
PFNWGLRELEASEPBUFFERDCEXTPROC __wglewReleasePbufferDCEXT = NULL;

PFNWGLCHOOSEPIXELFORMATEXTPROC __wglewChoosePixelFormatEXT = NULL;
PFNWGLGETPIXELFORMATATTRIBFVEXTPROC __wglewGetPixelFormatAttribfvEXT = NULL;
PFNWGLGETPIXELFORMATATTRIBIVEXTPROC __wglewGetPixelFormatAttribivEXT = NULL;

PFNWGLGETSWAPINTERVALEXTPROC __wglewGetSwapIntervalEXT = NULL;
PFNWGLSWAPINTERVALEXTPROC __wglewSwapIntervalEXT = NULL;

PFNWGLGETDIGITALVIDEOPARAMETERSI3DPROC __wglewGetDigitalVideoParametersI3D = NULL;
PFNWGLSETDIGITALVIDEOPARAMETERSI3DPROC __wglewSetDigitalVideoParametersI3D = NULL;

PFNWGLGETGAMMATABLEI3DPROC __wglewGetGammaTableI3D = NULL;
PFNWGLGETGAMMATABLEPARAMETERSI3DPROC __wglewGetGammaTableParametersI3D = NULL;
PFNWGLSETGAMMATABLEI3DPROC __wglewSetGammaTableI3D = NULL;
PFNWGLSETGAMMATABLEPARAMETERSI3DPROC __wglewSetGammaTableParametersI3D = NULL;

PFNWGLDISABLEGENLOCKI3DPROC __wglewDisableGenlockI3D = NULL;
PFNWGLENABLEGENLOCKI3DPROC __wglewEnableGenlockI3D = NULL;
PFNWGLGENLOCKSAMPLERATEI3DPROC __wglewGenlockSampleRateI3D = NULL;
PFNWGLGENLOCKSOURCEDELAYI3DPROC __wglewGenlockSourceDelayI3D = NULL;
PFNWGLGENLOCKSOURCEEDGEI3DPROC __wglewGenlockSourceEdgeI3D = NULL;
PFNWGLGENLOCKSOURCEI3DPROC __wglewGenlockSourceI3D = NULL;
PFNWGLGETGENLOCKSAMPLERATEI3DPROC __wglewGetGenlockSampleRateI3D = NULL;
PFNWGLGETGENLOCKSOURCEDELAYI3DPROC __wglewGetGenlockSourceDelayI3D = NULL;
PFNWGLGETGENLOCKSOURCEEDGEI3DPROC __wglewGetGenlockSourceEdgeI3D = NULL;
PFNWGLGETGENLOCKSOURCEI3DPROC __wglewGetGenlockSourceI3D = NULL;
PFNWGLISENABLEDGENLOCKI3DPROC __wglewIsEnabledGenlockI3D = NULL;
PFNWGLQUERYGENLOCKMAXSOURCEDELAYI3DPROC __wglewQueryGenlockMaxSourceDelayI3D = NULL;

PFNWGLASSOCIATEIMAGEBUFFEREVENTSI3DPROC __wglewAssociateImageBufferEventsI3D = NULL;
PFNWGLCREATEIMAGEBUFFERI3DPROC __wglewCreateImageBufferI3D = NULL;
PFNWGLDESTROYIMAGEBUFFERI3DPROC __wglewDestroyImageBufferI3D = NULL;
PFNWGLRELEASEIMAGEBUFFEREVENTSI3DPROC __wglewReleaseImageBufferEventsI3D = NULL;

PFNWGLDISABLEFRAMELOCKI3DPROC __wglewDisableFrameLockI3D = NULL;
PFNWGLENABLEFRAMELOCKI3DPROC __wglewEnableFrameLockI3D = NULL;
PFNWGLISENABLEDFRAMELOCKI3DPROC __wglewIsEnabledFrameLockI3D = NULL;
PFNWGLQUERYFRAMELOCKMASTERI3DPROC __wglewQueryFrameLockMasterI3D = NULL;

PFNWGLBEGINFRAMETRACKINGI3DPROC __wglewBeginFrameTrackingI3D = NULL;
PFNWGLENDFRAMETRACKINGI3DPROC __wglewEndFrameTrackingI3D = NULL;
PFNWGLGETFRAMEUSAGEI3DPROC __wglewGetFrameUsageI3D = NULL;
PFNWGLQUERYFRAMETRACKINGI3DPROC __wglewQueryFrameTrackingI3D = NULL;

PFNWGLDXCLOSEDEVICENVPROC __wglewDXCloseDeviceNV = NULL;
PFNWGLDXLOCKOBJECTSNVPROC __wglewDXLockObjectsNV = NULL;
PFNWGLDXOBJECTACCESSNVPROC __wglewDXObjectAccessNV = NULL;
PFNWGLDXOPENDEVICENVPROC __wglewDXOpenDeviceNV = NULL;
PFNWGLDXREGISTEROBJECTNVPROC __wglewDXRegisterObjectNV = NULL;
PFNWGLDXSETRESOURCESHAREHANDLENVPROC __wglewDXSetResourceShareHandleNV = NULL;
PFNWGLDXUNLOCKOBJECTSNVPROC __wglewDXUnlockObjectsNV = NULL;
PFNWGLDXUNREGISTEROBJECTNVPROC __wglewDXUnregisterObjectNV = NULL;

PFNWGLCOPYIMAGESUBDATANVPROC __wglewCopyImageSubDataNV = NULL;

PFNWGLDELAYBEFORESWAPNVPROC __wglewDelayBeforeSwapNV = NULL;

PFNWGLCREATEAFFINITYDCNVPROC __wglewCreateAffinityDCNV = NULL;
PFNWGLDELETEDCNVPROC __wglewDeleteDCNV = NULL;
PFNWGLENUMGPUDEVICESNVPROC __wglewEnumGpuDevicesNV = NULL;
PFNWGLENUMGPUSFROMAFFINITYDCNVPROC __wglewEnumGpusFromAffinityDCNV = NULL;
PFNWGLENUMGPUSNVPROC __wglewEnumGpusNV = NULL;

PFNWGLBINDVIDEODEVICENVPROC __wglewBindVideoDeviceNV = NULL;
PFNWGLENUMERATEVIDEODEVICESNVPROC __wglewEnumerateVideoDevicesNV = NULL;
PFNWGLQUERYCURRENTCONTEXTNVPROC __wglewQueryCurrentContextNV = NULL;

PFNWGLBINDSWAPBARRIERNVPROC __wglewBindSwapBarrierNV = NULL;
PFNWGLJOINSWAPGROUPNVPROC __wglewJoinSwapGroupNV = NULL;
PFNWGLQUERYFRAMECOUNTNVPROC __wglewQueryFrameCountNV = NULL;
PFNWGLQUERYMAXSWAPGROUPSNVPROC __wglewQueryMaxSwapGroupsNV = NULL;
PFNWGLQUERYSWAPGROUPNVPROC __wglewQuerySwapGroupNV = NULL;
PFNWGLRESETFRAMECOUNTNVPROC __wglewResetFrameCountNV = NULL;

PFNWGLALLOCATEMEMORYNVPROC __wglewAllocateMemoryNV = NULL;
PFNWGLFREEMEMORYNVPROC __wglewFreeMemoryNV = NULL;

PFNWGLBINDVIDEOCAPTUREDEVICENVPROC __wglewBindVideoCaptureDeviceNV = NULL;
PFNWGLENUMERATEVIDEOCAPTUREDEVICESNVPROC __wglewEnumerateVideoCaptureDevicesNV = NULL;
PFNWGLLOCKVIDEOCAPTUREDEVICENVPROC __wglewLockVideoCaptureDeviceNV = NULL;
PFNWGLQUERYVIDEOCAPTUREDEVICENVPROC __wglewQueryVideoCaptureDeviceNV = NULL;
PFNWGLRELEASEVIDEOCAPTUREDEVICENVPROC __wglewReleaseVideoCaptureDeviceNV = NULL;

PFNWGLBINDVIDEOIMAGENVPROC __wglewBindVideoImageNV = NULL;
PFNWGLGETVIDEODEVICENVPROC __wglewGetVideoDeviceNV = NULL;
PFNWGLGETVIDEOINFONVPROC __wglewGetVideoInfoNV = NULL;
PFNWGLRELEASEVIDEODEVICENVPROC __wglewReleaseVideoDeviceNV = NULL;
PFNWGLRELEASEVIDEOIMAGENVPROC __wglewReleaseVideoImageNV = NULL;
PFNWGLSENDPBUFFERTOVIDEONVPROC __wglewSendPbufferToVideoNV = NULL;

PFNWGLGETMSCRATEOMLPROC __wglewGetMscRateOML = NULL;
PFNWGLGETSYNCVALUESOMLPROC __wglewGetSyncValuesOML = NULL;
PFNWGLSWAPBUFFERSMSCOMLPROC __wglewSwapBuffersMscOML = NULL;
PFNWGLSWAPLAYERBUFFERSMSCOMLPROC __wglewSwapLayerBuffersMscOML = NULL;
PFNWGLWAITFORMSCOMLPROC __wglewWaitForMscOML = NULL;
PFNWGLWAITFORSBCOMLPROC __wglewWaitForSbcOML = NULL;
GLboolean __WGLEW_3DFX_multisample = GL_FALSE;
GLboolean __WGLEW_3DL_stereo_control = GL_FALSE;
GLboolean __WGLEW_AMD_gpu_association = GL_FALSE;
GLboolean __WGLEW_ARB_buffer_region = GL_FALSE;
GLboolean __WGLEW_ARB_context_flush_control = GL_FALSE;
GLboolean __WGLEW_ARB_create_context = GL_FALSE;
GLboolean __WGLEW_ARB_create_context_profile = GL_FALSE;
GLboolean __WGLEW_ARB_create_context_robustness = GL_FALSE;
GLboolean __WGLEW_ARB_extensions_string = GL_FALSE;
GLboolean __WGLEW_ARB_framebuffer_sRGB = GL_FALSE;
GLboolean __WGLEW_ARB_make_current_read = GL_FALSE;
GLboolean __WGLEW_ARB_multisample = GL_FALSE;
GLboolean __WGLEW_ARB_pbuffer = GL_FALSE;
GLboolean __WGLEW_ARB_pixel_format = GL_FALSE;
GLboolean __WGLEW_ARB_pixel_format_float = GL_FALSE;
GLboolean __WGLEW_ARB_render_texture = GL_FALSE;
GLboolean __WGLEW_ARB_robustness_application_isolation = GL_FALSE;
GLboolean __WGLEW_ARB_robustness_share_group_isolation = GL_FALSE;
GLboolean __WGLEW_ATI_pixel_format_float = GL_FALSE;
GLboolean __WGLEW_ATI_render_texture_rectangle = GL_FALSE;
GLboolean __WGLEW_EXT_create_context_es2_profile = GL_FALSE;
GLboolean __WGLEW_EXT_create_context_es_profile = GL_FALSE;
GLboolean __WGLEW_EXT_depth_float = GL_FALSE;
GLboolean __WGLEW_EXT_display_color_table = GL_FALSE;
GLboolean __WGLEW_EXT_extensions_string = GL_FALSE;
GLboolean __WGLEW_EXT_framebuffer_sRGB = GL_FALSE;
GLboolean __WGLEW_EXT_make_current_read = GL_FALSE;
GLboolean __WGLEW_EXT_multisample = GL_FALSE;
GLboolean __WGLEW_EXT_pbuffer = GL_FALSE;
GLboolean __WGLEW_EXT_pixel_format = GL_FALSE;
GLboolean __WGLEW_EXT_pixel_format_packed_float = GL_FALSE;
GLboolean __WGLEW_EXT_swap_control = GL_FALSE;
GLboolean __WGLEW_EXT_swap_control_tear = GL_FALSE;
GLboolean __WGLEW_I3D_digital_video_control = GL_FALSE;
GLboolean __WGLEW_I3D_gamma = GL_FALSE;
GLboolean __WGLEW_I3D_genlock = GL_FALSE;
GLboolean __WGLEW_I3D_image_buffer = GL_FALSE;
GLboolean __WGLEW_I3D_swap_frame_lock = GL_FALSE;
GLboolean __WGLEW_I3D_swap_frame_usage = GL_FALSE;
GLboolean __WGLEW_NV_DX_interop = GL_FALSE;
GLboolean __WGLEW_NV_DX_interop2 = GL_FALSE;
GLboolean __WGLEW_NV_copy_image = GL_FALSE;
GLboolean __WGLEW_NV_delay_before_swap = GL_FALSE;
GLboolean __WGLEW_NV_float_buffer = GL_FALSE;
GLboolean __WGLEW_NV_gpu_affinity = GL_FALSE;
GLboolean __WGLEW_NV_multisample_coverage = GL_FALSE;
GLboolean __WGLEW_NV_present_video = GL_FALSE;
GLboolean __WGLEW_NV_render_depth_texture = GL_FALSE;
GLboolean __WGLEW_NV_render_texture_rectangle = GL_FALSE;
GLboolean __WGLEW_NV_swap_group = GL_FALSE;
GLboolean __WGLEW_NV_vertex_array_range = GL_FALSE;
GLboolean __WGLEW_NV_video_capture = GL_FALSE;
GLboolean __WGLEW_NV_video_output = GL_FALSE;
GLboolean __WGLEW_OML_sync_control = GL_FALSE;

#endif /* !GLEW_MX */

#ifdef WGL_3DL_stereo_control

static GLboolean _glewInit_WGL_3DL_stereo_control (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglSetStereoEmitterState3DL = (PFNWGLSETSTEREOEMITTERSTATE3DLPROC)glewGetProcAddress((const GLubyte*)"wglSetStereoEmitterState3DL")) == NULL) || r;

  return r;
}

#endif /* WGL_3DL_stereo_control */

#ifdef WGL_AMD_gpu_association

static GLboolean _glewInit_WGL_AMD_gpu_association (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglBlitContextFramebufferAMD = (PFNWGLBLITCONTEXTFRAMEBUFFERAMDPROC)glewGetProcAddress((const GLubyte*)"wglBlitContextFramebufferAMD")) == NULL) || r;
  r = ((wglCreateAssociatedContextAMD = (PFNWGLCREATEASSOCIATEDCONTEXTAMDPROC)glewGetProcAddress((const GLubyte*)"wglCreateAssociatedContextAMD")) == NULL) || r;
  r = ((wglCreateAssociatedContextAttribsAMD = (PFNWGLCREATEASSOCIATEDCONTEXTATTRIBSAMDPROC)glewGetProcAddress((const GLubyte*)"wglCreateAssociatedContextAttribsAMD")) == NULL) || r;
  r = ((wglDeleteAssociatedContextAMD = (PFNWGLDELETEASSOCIATEDCONTEXTAMDPROC)glewGetProcAddress((const GLubyte*)"wglDeleteAssociatedContextAMD")) == NULL) || r;
  r = ((wglGetContextGPUIDAMD = (PFNWGLGETCONTEXTGPUIDAMDPROC)glewGetProcAddress((const GLubyte*)"wglGetContextGPUIDAMD")) == NULL) || r;
  r = ((wglGetCurrentAssociatedContextAMD = (PFNWGLGETCURRENTASSOCIATEDCONTEXTAMDPROC)glewGetProcAddress((const GLubyte*)"wglGetCurrentAssociatedContextAMD")) == NULL) || r;
  r = ((wglGetGPUIDsAMD = (PFNWGLGETGPUIDSAMDPROC)glewGetProcAddress((const GLubyte*)"wglGetGPUIDsAMD")) == NULL) || r;
  r = ((wglGetGPUInfoAMD = (PFNWGLGETGPUINFOAMDPROC)glewGetProcAddress((const GLubyte*)"wglGetGPUInfoAMD")) == NULL) || r;
  r = ((wglMakeAssociatedContextCurrentAMD = (PFNWGLMAKEASSOCIATEDCONTEXTCURRENTAMDPROC)glewGetProcAddress((const GLubyte*)"wglMakeAssociatedContextCurrentAMD")) == NULL) || r;

  return r;
}

#endif /* WGL_AMD_gpu_association */

#ifdef WGL_ARB_buffer_region

static GLboolean _glewInit_WGL_ARB_buffer_region (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglCreateBufferRegionARB = (PFNWGLCREATEBUFFERREGIONARBPROC)glewGetProcAddress((const GLubyte*)"wglCreateBufferRegionARB")) == NULL) || r;
  r = ((wglDeleteBufferRegionARB = (PFNWGLDELETEBUFFERREGIONARBPROC)glewGetProcAddress((const GLubyte*)"wglDeleteBufferRegionARB")) == NULL) || r;
  r = ((wglRestoreBufferRegionARB = (PFNWGLRESTOREBUFFERREGIONARBPROC)glewGetProcAddress((const GLubyte*)"wglRestoreBufferRegionARB")) == NULL) || r;
  r = ((wglSaveBufferRegionARB = (PFNWGLSAVEBUFFERREGIONARBPROC)glewGetProcAddress((const GLubyte*)"wglSaveBufferRegionARB")) == NULL) || r;

  return r;
}

#endif /* WGL_ARB_buffer_region */

#ifdef WGL_ARB_create_context

static GLboolean _glewInit_WGL_ARB_create_context (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)glewGetProcAddress((const GLubyte*)"wglCreateContextAttribsARB")) == NULL) || r;

  return r;
}

#endif /* WGL_ARB_create_context */

#ifdef WGL_ARB_extensions_string

static GLboolean _glewInit_WGL_ARB_extensions_string (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglGetExtensionsStringARB = (PFNWGLGETEXTENSIONSSTRINGARBPROC)glewGetProcAddress((const GLubyte*)"wglGetExtensionsStringARB")) == NULL) || r;

  return r;
}

#endif /* WGL_ARB_extensions_string */

#ifdef WGL_ARB_make_current_read

static GLboolean _glewInit_WGL_ARB_make_current_read (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglGetCurrentReadDCARB = (PFNWGLGETCURRENTREADDCARBPROC)glewGetProcAddress((const GLubyte*)"wglGetCurrentReadDCARB")) == NULL) || r;
  r = ((wglMakeContextCurrentARB = (PFNWGLMAKECONTEXTCURRENTARBPROC)glewGetProcAddress((const GLubyte*)"wglMakeContextCurrentARB")) == NULL) || r;

  return r;
}

#endif /* WGL_ARB_make_current_read */

#ifdef WGL_ARB_pbuffer

static GLboolean _glewInit_WGL_ARB_pbuffer (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglCreatePbufferARB = (PFNWGLCREATEPBUFFERARBPROC)glewGetProcAddress((const GLubyte*)"wglCreatePbufferARB")) == NULL) || r;
  r = ((wglDestroyPbufferARB = (PFNWGLDESTROYPBUFFERARBPROC)glewGetProcAddress((const GLubyte*)"wglDestroyPbufferARB")) == NULL) || r;
  r = ((wglGetPbufferDCARB = (PFNWGLGETPBUFFERDCARBPROC)glewGetProcAddress((const GLubyte*)"wglGetPbufferDCARB")) == NULL) || r;
  r = ((wglQueryPbufferARB = (PFNWGLQUERYPBUFFERARBPROC)glewGetProcAddress((const GLubyte*)"wglQueryPbufferARB")) == NULL) || r;
  r = ((wglReleasePbufferDCARB = (PFNWGLRELEASEPBUFFERDCARBPROC)glewGetProcAddress((const GLubyte*)"wglReleasePbufferDCARB")) == NULL) || r;

  return r;
}

#endif /* WGL_ARB_pbuffer */

#ifdef WGL_ARB_pixel_format

static GLboolean _glewInit_WGL_ARB_pixel_format (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglChoosePixelFormatARB = (PFNWGLCHOOSEPIXELFORMATARBPROC)glewGetProcAddress((const GLubyte*)"wglChoosePixelFormatARB")) == NULL) || r;
  r = ((wglGetPixelFormatAttribfvARB = (PFNWGLGETPIXELFORMATATTRIBFVARBPROC)glewGetProcAddress((const GLubyte*)"wglGetPixelFormatAttribfvARB")) == NULL) || r;
  r = ((wglGetPixelFormatAttribivARB = (PFNWGLGETPIXELFORMATATTRIBIVARBPROC)glewGetProcAddress((const GLubyte*)"wglGetPixelFormatAttribivARB")) == NULL) || r;

  return r;
}

#endif /* WGL_ARB_pixel_format */

#ifdef WGL_ARB_render_texture

static GLboolean _glewInit_WGL_ARB_render_texture (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglBindTexImageARB = (PFNWGLBINDTEXIMAGEARBPROC)glewGetProcAddress((const GLubyte*)"wglBindTexImageARB")) == NULL) || r;
  r = ((wglReleaseTexImageARB = (PFNWGLRELEASETEXIMAGEARBPROC)glewGetProcAddress((const GLubyte*)"wglReleaseTexImageARB")) == NULL) || r;
  r = ((wglSetPbufferAttribARB = (PFNWGLSETPBUFFERATTRIBARBPROC)glewGetProcAddress((const GLubyte*)"wglSetPbufferAttribARB")) == NULL) || r;

  return r;
}

#endif /* WGL_ARB_render_texture */

#ifdef WGL_EXT_display_color_table

static GLboolean _glewInit_WGL_EXT_display_color_table (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglBindDisplayColorTableEXT = (PFNWGLBINDDISPLAYCOLORTABLEEXTPROC)glewGetProcAddress((const GLubyte*)"wglBindDisplayColorTableEXT")) == NULL) || r;
  r = ((wglCreateDisplayColorTableEXT = (PFNWGLCREATEDISPLAYCOLORTABLEEXTPROC)glewGetProcAddress((const GLubyte*)"wglCreateDisplayColorTableEXT")) == NULL) || r;
  r = ((wglDestroyDisplayColorTableEXT = (PFNWGLDESTROYDISPLAYCOLORTABLEEXTPROC)glewGetProcAddress((const GLubyte*)"wglDestroyDisplayColorTableEXT")) == NULL) || r;
  r = ((wglLoadDisplayColorTableEXT = (PFNWGLLOADDISPLAYCOLORTABLEEXTPROC)glewGetProcAddress((const GLubyte*)"wglLoadDisplayColorTableEXT")) == NULL) || r;

  return r;
}

#endif /* WGL_EXT_display_color_table */

#ifdef WGL_EXT_extensions_string

static GLboolean _glewInit_WGL_EXT_extensions_string (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC)glewGetProcAddress((const GLubyte*)"wglGetExtensionsStringEXT")) == NULL) || r;

  return r;
}

#endif /* WGL_EXT_extensions_string */

#ifdef WGL_EXT_make_current_read

static GLboolean _glewInit_WGL_EXT_make_current_read (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglGetCurrentReadDCEXT = (PFNWGLGETCURRENTREADDCEXTPROC)glewGetProcAddress((const GLubyte*)"wglGetCurrentReadDCEXT")) == NULL) || r;
  r = ((wglMakeContextCurrentEXT = (PFNWGLMAKECONTEXTCURRENTEXTPROC)glewGetProcAddress((const GLubyte*)"wglMakeContextCurrentEXT")) == NULL) || r;

  return r;
}

#endif /* WGL_EXT_make_current_read */

#ifdef WGL_EXT_pbuffer

static GLboolean _glewInit_WGL_EXT_pbuffer (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglCreatePbufferEXT = (PFNWGLCREATEPBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"wglCreatePbufferEXT")) == NULL) || r;
  r = ((wglDestroyPbufferEXT = (PFNWGLDESTROYPBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"wglDestroyPbufferEXT")) == NULL) || r;
  r = ((wglGetPbufferDCEXT = (PFNWGLGETPBUFFERDCEXTPROC)glewGetProcAddress((const GLubyte*)"wglGetPbufferDCEXT")) == NULL) || r;
  r = ((wglQueryPbufferEXT = (PFNWGLQUERYPBUFFEREXTPROC)glewGetProcAddress((const GLubyte*)"wglQueryPbufferEXT")) == NULL) || r;
  r = ((wglReleasePbufferDCEXT = (PFNWGLRELEASEPBUFFERDCEXTPROC)glewGetProcAddress((const GLubyte*)"wglReleasePbufferDCEXT")) == NULL) || r;

  return r;
}

#endif /* WGL_EXT_pbuffer */

#ifdef WGL_EXT_pixel_format

static GLboolean _glewInit_WGL_EXT_pixel_format (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglChoosePixelFormatEXT = (PFNWGLCHOOSEPIXELFORMATEXTPROC)glewGetProcAddress((const GLubyte*)"wglChoosePixelFormatEXT")) == NULL) || r;
  r = ((wglGetPixelFormatAttribfvEXT = (PFNWGLGETPIXELFORMATATTRIBFVEXTPROC)glewGetProcAddress((const GLubyte*)"wglGetPixelFormatAttribfvEXT")) == NULL) || r;
  r = ((wglGetPixelFormatAttribivEXT = (PFNWGLGETPIXELFORMATATTRIBIVEXTPROC)glewGetProcAddress((const GLubyte*)"wglGetPixelFormatAttribivEXT")) == NULL) || r;

  return r;
}

#endif /* WGL_EXT_pixel_format */

#ifdef WGL_EXT_swap_control

static GLboolean _glewInit_WGL_EXT_swap_control (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglGetSwapIntervalEXT = (PFNWGLGETSWAPINTERVALEXTPROC)glewGetProcAddress((const GLubyte*)"wglGetSwapIntervalEXT")) == NULL) || r;
  r = ((wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)glewGetProcAddress((const GLubyte*)"wglSwapIntervalEXT")) == NULL) || r;

  return r;
}

#endif /* WGL_EXT_swap_control */

#ifdef WGL_I3D_digital_video_control

static GLboolean _glewInit_WGL_I3D_digital_video_control (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglGetDigitalVideoParametersI3D = (PFNWGLGETDIGITALVIDEOPARAMETERSI3DPROC)glewGetProcAddress((const GLubyte*)"wglGetDigitalVideoParametersI3D")) == NULL) || r;
  r = ((wglSetDigitalVideoParametersI3D = (PFNWGLSETDIGITALVIDEOPARAMETERSI3DPROC)glewGetProcAddress((const GLubyte*)"wglSetDigitalVideoParametersI3D")) == NULL) || r;

  return r;
}

#endif /* WGL_I3D_digital_video_control */

#ifdef WGL_I3D_gamma

static GLboolean _glewInit_WGL_I3D_gamma (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglGetGammaTableI3D = (PFNWGLGETGAMMATABLEI3DPROC)glewGetProcAddress((const GLubyte*)"wglGetGammaTableI3D")) == NULL) || r;
  r = ((wglGetGammaTableParametersI3D = (PFNWGLGETGAMMATABLEPARAMETERSI3DPROC)glewGetProcAddress((const GLubyte*)"wglGetGammaTableParametersI3D")) == NULL) || r;
  r = ((wglSetGammaTableI3D = (PFNWGLSETGAMMATABLEI3DPROC)glewGetProcAddress((const GLubyte*)"wglSetGammaTableI3D")) == NULL) || r;
  r = ((wglSetGammaTableParametersI3D = (PFNWGLSETGAMMATABLEPARAMETERSI3DPROC)glewGetProcAddress((const GLubyte*)"wglSetGammaTableParametersI3D")) == NULL) || r;

  return r;
}

#endif /* WGL_I3D_gamma */

#ifdef WGL_I3D_genlock

static GLboolean _glewInit_WGL_I3D_genlock (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglDisableGenlockI3D = (PFNWGLDISABLEGENLOCKI3DPROC)glewGetProcAddress((const GLubyte*)"wglDisableGenlockI3D")) == NULL) || r;
  r = ((wglEnableGenlockI3D = (PFNWGLENABLEGENLOCKI3DPROC)glewGetProcAddress((const GLubyte*)"wglEnableGenlockI3D")) == NULL) || r;
  r = ((wglGenlockSampleRateI3D = (PFNWGLGENLOCKSAMPLERATEI3DPROC)glewGetProcAddress((const GLubyte*)"wglGenlockSampleRateI3D")) == NULL) || r;
  r = ((wglGenlockSourceDelayI3D = (PFNWGLGENLOCKSOURCEDELAYI3DPROC)glewGetProcAddress((const GLubyte*)"wglGenlockSourceDelayI3D")) == NULL) || r;
  r = ((wglGenlockSourceEdgeI3D = (PFNWGLGENLOCKSOURCEEDGEI3DPROC)glewGetProcAddress((const GLubyte*)"wglGenlockSourceEdgeI3D")) == NULL) || r;
  r = ((wglGenlockSourceI3D = (PFNWGLGENLOCKSOURCEI3DPROC)glewGetProcAddress((const GLubyte*)"wglGenlockSourceI3D")) == NULL) || r;
  r = ((wglGetGenlockSampleRateI3D = (PFNWGLGETGENLOCKSAMPLERATEI3DPROC)glewGetProcAddress((const GLubyte*)"wglGetGenlockSampleRateI3D")) == NULL) || r;
  r = ((wglGetGenlockSourceDelayI3D = (PFNWGLGETGENLOCKSOURCEDELAYI3DPROC)glewGetProcAddress((const GLubyte*)"wglGetGenlockSourceDelayI3D")) == NULL) || r;
  r = ((wglGetGenlockSourceEdgeI3D = (PFNWGLGETGENLOCKSOURCEEDGEI3DPROC)glewGetProcAddress((const GLubyte*)"wglGetGenlockSourceEdgeI3D")) == NULL) || r;
  r = ((wglGetGenlockSourceI3D = (PFNWGLGETGENLOCKSOURCEI3DPROC)glewGetProcAddress((const GLubyte*)"wglGetGenlockSourceI3D")) == NULL) || r;
  r = ((wglIsEnabledGenlockI3D = (PFNWGLISENABLEDGENLOCKI3DPROC)glewGetProcAddress((const GLubyte*)"wglIsEnabledGenlockI3D")) == NULL) || r;
  r = ((wglQueryGenlockMaxSourceDelayI3D = (PFNWGLQUERYGENLOCKMAXSOURCEDELAYI3DPROC)glewGetProcAddress((const GLubyte*)"wglQueryGenlockMaxSourceDelayI3D")) == NULL) || r;

  return r;
}

#endif /* WGL_I3D_genlock */

#ifdef WGL_I3D_image_buffer

static GLboolean _glewInit_WGL_I3D_image_buffer (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglAssociateImageBufferEventsI3D = (PFNWGLASSOCIATEIMAGEBUFFEREVENTSI3DPROC)glewGetProcAddress((const GLubyte*)"wglAssociateImageBufferEventsI3D")) == NULL) || r;
  r = ((wglCreateImageBufferI3D = (PFNWGLCREATEIMAGEBUFFERI3DPROC)glewGetProcAddress((const GLubyte*)"wglCreateImageBufferI3D")) == NULL) || r;
  r = ((wglDestroyImageBufferI3D = (PFNWGLDESTROYIMAGEBUFFERI3DPROC)glewGetProcAddress((const GLubyte*)"wglDestroyImageBufferI3D")) == NULL) || r;
  r = ((wglReleaseImageBufferEventsI3D = (PFNWGLRELEASEIMAGEBUFFEREVENTSI3DPROC)glewGetProcAddress((const GLubyte*)"wglReleaseImageBufferEventsI3D")) == NULL) || r;

  return r;
}

#endif /* WGL_I3D_image_buffer */

#ifdef WGL_I3D_swap_frame_lock

static GLboolean _glewInit_WGL_I3D_swap_frame_lock (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglDisableFrameLockI3D = (PFNWGLDISABLEFRAMELOCKI3DPROC)glewGetProcAddress((const GLubyte*)"wglDisableFrameLockI3D")) == NULL) || r;
  r = ((wglEnableFrameLockI3D = (PFNWGLENABLEFRAMELOCKI3DPROC)glewGetProcAddress((const GLubyte*)"wglEnableFrameLockI3D")) == NULL) || r;
  r = ((wglIsEnabledFrameLockI3D = (PFNWGLISENABLEDFRAMELOCKI3DPROC)glewGetProcAddress((const GLubyte*)"wglIsEnabledFrameLockI3D")) == NULL) || r;
  r = ((wglQueryFrameLockMasterI3D = (PFNWGLQUERYFRAMELOCKMASTERI3DPROC)glewGetProcAddress((const GLubyte*)"wglQueryFrameLockMasterI3D")) == NULL) || r;

  return r;
}

#endif /* WGL_I3D_swap_frame_lock */

#ifdef WGL_I3D_swap_frame_usage

static GLboolean _glewInit_WGL_I3D_swap_frame_usage (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglBeginFrameTrackingI3D = (PFNWGLBEGINFRAMETRACKINGI3DPROC)glewGetProcAddress((const GLubyte*)"wglBeginFrameTrackingI3D")) == NULL) || r;
  r = ((wglEndFrameTrackingI3D = (PFNWGLENDFRAMETRACKINGI3DPROC)glewGetProcAddress((const GLubyte*)"wglEndFrameTrackingI3D")) == NULL) || r;
  r = ((wglGetFrameUsageI3D = (PFNWGLGETFRAMEUSAGEI3DPROC)glewGetProcAddress((const GLubyte*)"wglGetFrameUsageI3D")) == NULL) || r;
  r = ((wglQueryFrameTrackingI3D = (PFNWGLQUERYFRAMETRACKINGI3DPROC)glewGetProcAddress((const GLubyte*)"wglQueryFrameTrackingI3D")) == NULL) || r;

  return r;
}

#endif /* WGL_I3D_swap_frame_usage */

#ifdef WGL_NV_DX_interop

static GLboolean _glewInit_WGL_NV_DX_interop (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglDXCloseDeviceNV = (PFNWGLDXCLOSEDEVICENVPROC)glewGetProcAddress((const GLubyte*)"wglDXCloseDeviceNV")) == NULL) || r;
  r = ((wglDXLockObjectsNV = (PFNWGLDXLOCKOBJECTSNVPROC)glewGetProcAddress((const GLubyte*)"wglDXLockObjectsNV")) == NULL) || r;
  r = ((wglDXObjectAccessNV = (PFNWGLDXOBJECTACCESSNVPROC)glewGetProcAddress((const GLubyte*)"wglDXObjectAccessNV")) == NULL) || r;
  r = ((wglDXOpenDeviceNV = (PFNWGLDXOPENDEVICENVPROC)glewGetProcAddress((const GLubyte*)"wglDXOpenDeviceNV")) == NULL) || r;
  r = ((wglDXRegisterObjectNV = (PFNWGLDXREGISTEROBJECTNVPROC)glewGetProcAddress((const GLubyte*)"wglDXRegisterObjectNV")) == NULL) || r;
  r = ((wglDXSetResourceShareHandleNV = (PFNWGLDXSETRESOURCESHAREHANDLENVPROC)glewGetProcAddress((const GLubyte*)"wglDXSetResourceShareHandleNV")) == NULL) || r;
  r = ((wglDXUnlockObjectsNV = (PFNWGLDXUNLOCKOBJECTSNVPROC)glewGetProcAddress((const GLubyte*)"wglDXUnlockObjectsNV")) == NULL) || r;
  r = ((wglDXUnregisterObjectNV = (PFNWGLDXUNREGISTEROBJECTNVPROC)glewGetProcAddress((const GLubyte*)"wglDXUnregisterObjectNV")) == NULL) || r;

  return r;
}

#endif /* WGL_NV_DX_interop */

#ifdef WGL_NV_copy_image

static GLboolean _glewInit_WGL_NV_copy_image (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglCopyImageSubDataNV = (PFNWGLCOPYIMAGESUBDATANVPROC)glewGetProcAddress((const GLubyte*)"wglCopyImageSubDataNV")) == NULL) || r;

  return r;
}

#endif /* WGL_NV_copy_image */

#ifdef WGL_NV_delay_before_swap

static GLboolean _glewInit_WGL_NV_delay_before_swap (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglDelayBeforeSwapNV = (PFNWGLDELAYBEFORESWAPNVPROC)glewGetProcAddress((const GLubyte*)"wglDelayBeforeSwapNV")) == NULL) || r;

  return r;
}

#endif /* WGL_NV_delay_before_swap */

#ifdef WGL_NV_gpu_affinity

static GLboolean _glewInit_WGL_NV_gpu_affinity (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglCreateAffinityDCNV = (PFNWGLCREATEAFFINITYDCNVPROC)glewGetProcAddress((const GLubyte*)"wglCreateAffinityDCNV")) == NULL) || r;
  r = ((wglDeleteDCNV = (PFNWGLDELETEDCNVPROC)glewGetProcAddress((const GLubyte*)"wglDeleteDCNV")) == NULL) || r;
  r = ((wglEnumGpuDevicesNV = (PFNWGLENUMGPUDEVICESNVPROC)glewGetProcAddress((const GLubyte*)"wglEnumGpuDevicesNV")) == NULL) || r;
  r = ((wglEnumGpusFromAffinityDCNV = (PFNWGLENUMGPUSFROMAFFINITYDCNVPROC)glewGetProcAddress((const GLubyte*)"wglEnumGpusFromAffinityDCNV")) == NULL) || r;
  r = ((wglEnumGpusNV = (PFNWGLENUMGPUSNVPROC)glewGetProcAddress((const GLubyte*)"wglEnumGpusNV")) == NULL) || r;

  return r;
}

#endif /* WGL_NV_gpu_affinity */

#ifdef WGL_NV_present_video

static GLboolean _glewInit_WGL_NV_present_video (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglBindVideoDeviceNV = (PFNWGLBINDVIDEODEVICENVPROC)glewGetProcAddress((const GLubyte*)"wglBindVideoDeviceNV")) == NULL) || r;
  r = ((wglEnumerateVideoDevicesNV = (PFNWGLENUMERATEVIDEODEVICESNVPROC)glewGetProcAddress((const GLubyte*)"wglEnumerateVideoDevicesNV")) == NULL) || r;
  r = ((wglQueryCurrentContextNV = (PFNWGLQUERYCURRENTCONTEXTNVPROC)glewGetProcAddress((const GLubyte*)"wglQueryCurrentContextNV")) == NULL) || r;

  return r;
}

#endif /* WGL_NV_present_video */

#ifdef WGL_NV_swap_group

static GLboolean _glewInit_WGL_NV_swap_group (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglBindSwapBarrierNV = (PFNWGLBINDSWAPBARRIERNVPROC)glewGetProcAddress((const GLubyte*)"wglBindSwapBarrierNV")) == NULL) || r;
  r = ((wglJoinSwapGroupNV = (PFNWGLJOINSWAPGROUPNVPROC)glewGetProcAddress((const GLubyte*)"wglJoinSwapGroupNV")) == NULL) || r;
  r = ((wglQueryFrameCountNV = (PFNWGLQUERYFRAMECOUNTNVPROC)glewGetProcAddress((const GLubyte*)"wglQueryFrameCountNV")) == NULL) || r;
  r = ((wglQueryMaxSwapGroupsNV = (PFNWGLQUERYMAXSWAPGROUPSNVPROC)glewGetProcAddress((const GLubyte*)"wglQueryMaxSwapGroupsNV")) == NULL) || r;
  r = ((wglQuerySwapGroupNV = (PFNWGLQUERYSWAPGROUPNVPROC)glewGetProcAddress((const GLubyte*)"wglQuerySwapGroupNV")) == NULL) || r;
  r = ((wglResetFrameCountNV = (PFNWGLRESETFRAMECOUNTNVPROC)glewGetProcAddress((const GLubyte*)"wglResetFrameCountNV")) == NULL) || r;

  return r;
}

#endif /* WGL_NV_swap_group */

#ifdef WGL_NV_vertex_array_range

static GLboolean _glewInit_WGL_NV_vertex_array_range (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglAllocateMemoryNV = (PFNWGLALLOCATEMEMORYNVPROC)glewGetProcAddress((const GLubyte*)"wglAllocateMemoryNV")) == NULL) || r;
  r = ((wglFreeMemoryNV = (PFNWGLFREEMEMORYNVPROC)glewGetProcAddress((const GLubyte*)"wglFreeMemoryNV")) == NULL) || r;

  return r;
}

#endif /* WGL_NV_vertex_array_range */

#ifdef WGL_NV_video_capture

static GLboolean _glewInit_WGL_NV_video_capture (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglBindVideoCaptureDeviceNV = (PFNWGLBINDVIDEOCAPTUREDEVICENVPROC)glewGetProcAddress((const GLubyte*)"wglBindVideoCaptureDeviceNV")) == NULL) || r;
  r = ((wglEnumerateVideoCaptureDevicesNV = (PFNWGLENUMERATEVIDEOCAPTUREDEVICESNVPROC)glewGetProcAddress((const GLubyte*)"wglEnumerateVideoCaptureDevicesNV")) == NULL) || r;
  r = ((wglLockVideoCaptureDeviceNV = (PFNWGLLOCKVIDEOCAPTUREDEVICENVPROC)glewGetProcAddress((const GLubyte*)"wglLockVideoCaptureDeviceNV")) == NULL) || r;
  r = ((wglQueryVideoCaptureDeviceNV = (PFNWGLQUERYVIDEOCAPTUREDEVICENVPROC)glewGetProcAddress((const GLubyte*)"wglQueryVideoCaptureDeviceNV")) == NULL) || r;
  r = ((wglReleaseVideoCaptureDeviceNV = (PFNWGLRELEASEVIDEOCAPTUREDEVICENVPROC)glewGetProcAddress((const GLubyte*)"wglReleaseVideoCaptureDeviceNV")) == NULL) || r;

  return r;
}

#endif /* WGL_NV_video_capture */

#ifdef WGL_NV_video_output

static GLboolean _glewInit_WGL_NV_video_output (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglBindVideoImageNV = (PFNWGLBINDVIDEOIMAGENVPROC)glewGetProcAddress((const GLubyte*)"wglBindVideoImageNV")) == NULL) || r;
  r = ((wglGetVideoDeviceNV = (PFNWGLGETVIDEODEVICENVPROC)glewGetProcAddress((const GLubyte*)"wglGetVideoDeviceNV")) == NULL) || r;
  r = ((wglGetVideoInfoNV = (PFNWGLGETVIDEOINFONVPROC)glewGetProcAddress((const GLubyte*)"wglGetVideoInfoNV")) == NULL) || r;
  r = ((wglReleaseVideoDeviceNV = (PFNWGLRELEASEVIDEODEVICENVPROC)glewGetProcAddress((const GLubyte*)"wglReleaseVideoDeviceNV")) == NULL) || r;
  r = ((wglReleaseVideoImageNV = (PFNWGLRELEASEVIDEOIMAGENVPROC)glewGetProcAddress((const GLubyte*)"wglReleaseVideoImageNV")) == NULL) || r;
  r = ((wglSendPbufferToVideoNV = (PFNWGLSENDPBUFFERTOVIDEONVPROC)glewGetProcAddress((const GLubyte*)"wglSendPbufferToVideoNV")) == NULL) || r;

  return r;
}

#endif /* WGL_NV_video_output */

#ifdef WGL_OML_sync_control

static GLboolean _glewInit_WGL_OML_sync_control (WGLEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((wglGetMscRateOML = (PFNWGLGETMSCRATEOMLPROC)glewGetProcAddress((const GLubyte*)"wglGetMscRateOML")) == NULL) || r;
  r = ((wglGetSyncValuesOML = (PFNWGLGETSYNCVALUESOMLPROC)glewGetProcAddress((const GLubyte*)"wglGetSyncValuesOML")) == NULL) || r;
  r = ((wglSwapBuffersMscOML = (PFNWGLSWAPBUFFERSMSCOMLPROC)glewGetProcAddress((const GLubyte*)"wglSwapBuffersMscOML")) == NULL) || r;
  r = ((wglSwapLayerBuffersMscOML = (PFNWGLSWAPLAYERBUFFERSMSCOMLPROC)glewGetProcAddress((const GLubyte*)"wglSwapLayerBuffersMscOML")) == NULL) || r;
  r = ((wglWaitForMscOML = (PFNWGLWAITFORMSCOMLPROC)glewGetProcAddress((const GLubyte*)"wglWaitForMscOML")) == NULL) || r;
  r = ((wglWaitForSbcOML = (PFNWGLWAITFORSBCOMLPROC)glewGetProcAddress((const GLubyte*)"wglWaitForSbcOML")) == NULL) || r;

  return r;
}

#endif /* WGL_OML_sync_control */

/* ------------------------------------------------------------------------- */

static PFNWGLGETEXTENSIONSSTRINGARBPROC _wglewGetExtensionsStringARB = NULL;
static PFNWGLGETEXTENSIONSSTRINGEXTPROC _wglewGetExtensionsStringEXT = NULL;

GLboolean GLEWAPIENTRY wglewGetExtension (const char* name)
{    
  const GLubyte* start;
  const GLubyte* end;
  if (_wglewGetExtensionsStringARB == NULL)
    if (_wglewGetExtensionsStringEXT == NULL)
      return GL_FALSE;
    else
      start = (const GLubyte*)_wglewGetExtensionsStringEXT();
  else
    start = (const GLubyte*)_wglewGetExtensionsStringARB(wglGetCurrentDC());
  if (start == 0)
    return GL_FALSE;
  end = start + _glewStrLen(start);
  return _glewSearchExtension(name, start, end);
}

#ifdef GLEW_MX
GLenum GLEWAPIENTRY wglewContextInit (WGLEW_CONTEXT_ARG_DEF_LIST)
#else
GLenum GLEWAPIENTRY wglewInit (WGLEW_CONTEXT_ARG_DEF_LIST)
#endif
{
  GLboolean crippled;
  const GLubyte* extStart;
  const GLubyte* extEnd;
  /* find wgl extension string query functions */
  _wglewGetExtensionsStringARB = (PFNWGLGETEXTENSIONSSTRINGARBPROC)glewGetProcAddress((const GLubyte*)"wglGetExtensionsStringARB");
  _wglewGetExtensionsStringEXT = (PFNWGLGETEXTENSIONSSTRINGEXTPROC)glewGetProcAddress((const GLubyte*)"wglGetExtensionsStringEXT");
  /* query wgl extension string */
  if (_wglewGetExtensionsStringARB == NULL)
    if (_wglewGetExtensionsStringEXT == NULL)
      extStart = (const GLubyte*)"";
    else
      extStart = (const GLubyte*)_wglewGetExtensionsStringEXT();
  else
    extStart = (const GLubyte*)_wglewGetExtensionsStringARB(wglGetCurrentDC());
  extEnd = extStart + _glewStrLen(extStart);
  /* initialize extensions */
  crippled = _wglewGetExtensionsStringARB == NULL && _wglewGetExtensionsStringEXT == NULL;
#ifdef WGL_3DFX_multisample
  WGLEW_3DFX_multisample = _glewSearchExtension("WGL_3DFX_multisample", extStart, extEnd);
#endif /* WGL_3DFX_multisample */
#ifdef WGL_3DL_stereo_control
  WGLEW_3DL_stereo_control = _glewSearchExtension("WGL_3DL_stereo_control", extStart, extEnd);
  if (glewExperimental || WGLEW_3DL_stereo_control|| crippled) WGLEW_3DL_stereo_control= !_glewInit_WGL_3DL_stereo_control(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_3DL_stereo_control */
#ifdef WGL_AMD_gpu_association
  WGLEW_AMD_gpu_association = _glewSearchExtension("WGL_AMD_gpu_association", extStart, extEnd);
  if (glewExperimental || WGLEW_AMD_gpu_association|| crippled) WGLEW_AMD_gpu_association= !_glewInit_WGL_AMD_gpu_association(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_AMD_gpu_association */
#ifdef WGL_ARB_buffer_region
  WGLEW_ARB_buffer_region = _glewSearchExtension("WGL_ARB_buffer_region", extStart, extEnd);
  if (glewExperimental || WGLEW_ARB_buffer_region|| crippled) WGLEW_ARB_buffer_region= !_glewInit_WGL_ARB_buffer_region(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_ARB_buffer_region */
#ifdef WGL_ARB_context_flush_control
  WGLEW_ARB_context_flush_control = _glewSearchExtension("WGL_ARB_context_flush_control", extStart, extEnd);
#endif /* WGL_ARB_context_flush_control */
#ifdef WGL_ARB_create_context
  WGLEW_ARB_create_context = _glewSearchExtension("WGL_ARB_create_context", extStart, extEnd);
  if (glewExperimental || WGLEW_ARB_create_context|| crippled) WGLEW_ARB_create_context= !_glewInit_WGL_ARB_create_context(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_ARB_create_context */
#ifdef WGL_ARB_create_context_profile
  WGLEW_ARB_create_context_profile = _glewSearchExtension("WGL_ARB_create_context_profile", extStart, extEnd);
#endif /* WGL_ARB_create_context_profile */
#ifdef WGL_ARB_create_context_robustness
  WGLEW_ARB_create_context_robustness = _glewSearchExtension("WGL_ARB_create_context_robustness", extStart, extEnd);
#endif /* WGL_ARB_create_context_robustness */
#ifdef WGL_ARB_extensions_string
  WGLEW_ARB_extensions_string = _glewSearchExtension("WGL_ARB_extensions_string", extStart, extEnd);
  if (glewExperimental || WGLEW_ARB_extensions_string|| crippled) WGLEW_ARB_extensions_string= !_glewInit_WGL_ARB_extensions_string(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_ARB_extensions_string */
#ifdef WGL_ARB_framebuffer_sRGB
  WGLEW_ARB_framebuffer_sRGB = _glewSearchExtension("WGL_ARB_framebuffer_sRGB", extStart, extEnd);
#endif /* WGL_ARB_framebuffer_sRGB */
#ifdef WGL_ARB_make_current_read
  WGLEW_ARB_make_current_read = _glewSearchExtension("WGL_ARB_make_current_read", extStart, extEnd);
  if (glewExperimental || WGLEW_ARB_make_current_read|| crippled) WGLEW_ARB_make_current_read= !_glewInit_WGL_ARB_make_current_read(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_ARB_make_current_read */
#ifdef WGL_ARB_multisample
  WGLEW_ARB_multisample = _glewSearchExtension("WGL_ARB_multisample", extStart, extEnd);
#endif /* WGL_ARB_multisample */
#ifdef WGL_ARB_pbuffer
  WGLEW_ARB_pbuffer = _glewSearchExtension("WGL_ARB_pbuffer", extStart, extEnd);
  if (glewExperimental || WGLEW_ARB_pbuffer|| crippled) WGLEW_ARB_pbuffer= !_glewInit_WGL_ARB_pbuffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_ARB_pbuffer */
#ifdef WGL_ARB_pixel_format
  WGLEW_ARB_pixel_format = _glewSearchExtension("WGL_ARB_pixel_format", extStart, extEnd);
  if (glewExperimental || WGLEW_ARB_pixel_format|| crippled) WGLEW_ARB_pixel_format= !_glewInit_WGL_ARB_pixel_format(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_ARB_pixel_format */
#ifdef WGL_ARB_pixel_format_float
  WGLEW_ARB_pixel_format_float = _glewSearchExtension("WGL_ARB_pixel_format_float", extStart, extEnd);
#endif /* WGL_ARB_pixel_format_float */
#ifdef WGL_ARB_render_texture
  WGLEW_ARB_render_texture = _glewSearchExtension("WGL_ARB_render_texture", extStart, extEnd);
  if (glewExperimental || WGLEW_ARB_render_texture|| crippled) WGLEW_ARB_render_texture= !_glewInit_WGL_ARB_render_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_ARB_render_texture */
#ifdef WGL_ARB_robustness_application_isolation
  WGLEW_ARB_robustness_application_isolation = _glewSearchExtension("WGL_ARB_robustness_application_isolation", extStart, extEnd);
#endif /* WGL_ARB_robustness_application_isolation */
#ifdef WGL_ARB_robustness_share_group_isolation
  WGLEW_ARB_robustness_share_group_isolation = _glewSearchExtension("WGL_ARB_robustness_share_group_isolation", extStart, extEnd);
#endif /* WGL_ARB_robustness_share_group_isolation */
#ifdef WGL_ATI_pixel_format_float
  WGLEW_ATI_pixel_format_float = _glewSearchExtension("WGL_ATI_pixel_format_float", extStart, extEnd);
#endif /* WGL_ATI_pixel_format_float */
#ifdef WGL_ATI_render_texture_rectangle
  WGLEW_ATI_render_texture_rectangle = _glewSearchExtension("WGL_ATI_render_texture_rectangle", extStart, extEnd);
#endif /* WGL_ATI_render_texture_rectangle */
#ifdef WGL_EXT_create_context_es2_profile
  WGLEW_EXT_create_context_es2_profile = _glewSearchExtension("WGL_EXT_create_context_es2_profile", extStart, extEnd);
#endif /* WGL_EXT_create_context_es2_profile */
#ifdef WGL_EXT_create_context_es_profile
  WGLEW_EXT_create_context_es_profile = _glewSearchExtension("WGL_EXT_create_context_es_profile", extStart, extEnd);
#endif /* WGL_EXT_create_context_es_profile */
#ifdef WGL_EXT_depth_float
  WGLEW_EXT_depth_float = _glewSearchExtension("WGL_EXT_depth_float", extStart, extEnd);
#endif /* WGL_EXT_depth_float */
#ifdef WGL_EXT_display_color_table
  WGLEW_EXT_display_color_table = _glewSearchExtension("WGL_EXT_display_color_table", extStart, extEnd);
  if (glewExperimental || WGLEW_EXT_display_color_table|| crippled) WGLEW_EXT_display_color_table= !_glewInit_WGL_EXT_display_color_table(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_EXT_display_color_table */
#ifdef WGL_EXT_extensions_string
  WGLEW_EXT_extensions_string = _glewSearchExtension("WGL_EXT_extensions_string", extStart, extEnd);
  if (glewExperimental || WGLEW_EXT_extensions_string|| crippled) WGLEW_EXT_extensions_string= !_glewInit_WGL_EXT_extensions_string(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_EXT_extensions_string */
#ifdef WGL_EXT_framebuffer_sRGB
  WGLEW_EXT_framebuffer_sRGB = _glewSearchExtension("WGL_EXT_framebuffer_sRGB", extStart, extEnd);
#endif /* WGL_EXT_framebuffer_sRGB */
#ifdef WGL_EXT_make_current_read
  WGLEW_EXT_make_current_read = _glewSearchExtension("WGL_EXT_make_current_read", extStart, extEnd);
  if (glewExperimental || WGLEW_EXT_make_current_read|| crippled) WGLEW_EXT_make_current_read= !_glewInit_WGL_EXT_make_current_read(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_EXT_make_current_read */
#ifdef WGL_EXT_multisample
  WGLEW_EXT_multisample = _glewSearchExtension("WGL_EXT_multisample", extStart, extEnd);
#endif /* WGL_EXT_multisample */
#ifdef WGL_EXT_pbuffer
  WGLEW_EXT_pbuffer = _glewSearchExtension("WGL_EXT_pbuffer", extStart, extEnd);
  if (glewExperimental || WGLEW_EXT_pbuffer|| crippled) WGLEW_EXT_pbuffer= !_glewInit_WGL_EXT_pbuffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_EXT_pbuffer */
#ifdef WGL_EXT_pixel_format
  WGLEW_EXT_pixel_format = _glewSearchExtension("WGL_EXT_pixel_format", extStart, extEnd);
  if (glewExperimental || WGLEW_EXT_pixel_format|| crippled) WGLEW_EXT_pixel_format= !_glewInit_WGL_EXT_pixel_format(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_EXT_pixel_format */
#ifdef WGL_EXT_pixel_format_packed_float
  WGLEW_EXT_pixel_format_packed_float = _glewSearchExtension("WGL_EXT_pixel_format_packed_float", extStart, extEnd);
#endif /* WGL_EXT_pixel_format_packed_float */
#ifdef WGL_EXT_swap_control
  WGLEW_EXT_swap_control = _glewSearchExtension("WGL_EXT_swap_control", extStart, extEnd);
  if (glewExperimental || WGLEW_EXT_swap_control|| crippled) WGLEW_EXT_swap_control= !_glewInit_WGL_EXT_swap_control(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_EXT_swap_control */
#ifdef WGL_EXT_swap_control_tear
  WGLEW_EXT_swap_control_tear = _glewSearchExtension("WGL_EXT_swap_control_tear", extStart, extEnd);
#endif /* WGL_EXT_swap_control_tear */
#ifdef WGL_I3D_digital_video_control
  WGLEW_I3D_digital_video_control = _glewSearchExtension("WGL_I3D_digital_video_control", extStart, extEnd);
  if (glewExperimental || WGLEW_I3D_digital_video_control|| crippled) WGLEW_I3D_digital_video_control= !_glewInit_WGL_I3D_digital_video_control(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_I3D_digital_video_control */
#ifdef WGL_I3D_gamma
  WGLEW_I3D_gamma = _glewSearchExtension("WGL_I3D_gamma", extStart, extEnd);
  if (glewExperimental || WGLEW_I3D_gamma|| crippled) WGLEW_I3D_gamma= !_glewInit_WGL_I3D_gamma(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_I3D_gamma */
#ifdef WGL_I3D_genlock
  WGLEW_I3D_genlock = _glewSearchExtension("WGL_I3D_genlock", extStart, extEnd);
  if (glewExperimental || WGLEW_I3D_genlock|| crippled) WGLEW_I3D_genlock= !_glewInit_WGL_I3D_genlock(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_I3D_genlock */
#ifdef WGL_I3D_image_buffer
  WGLEW_I3D_image_buffer = _glewSearchExtension("WGL_I3D_image_buffer", extStart, extEnd);
  if (glewExperimental || WGLEW_I3D_image_buffer|| crippled) WGLEW_I3D_image_buffer= !_glewInit_WGL_I3D_image_buffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_I3D_image_buffer */
#ifdef WGL_I3D_swap_frame_lock
  WGLEW_I3D_swap_frame_lock = _glewSearchExtension("WGL_I3D_swap_frame_lock", extStart, extEnd);
  if (glewExperimental || WGLEW_I3D_swap_frame_lock|| crippled) WGLEW_I3D_swap_frame_lock= !_glewInit_WGL_I3D_swap_frame_lock(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_I3D_swap_frame_lock */
#ifdef WGL_I3D_swap_frame_usage
  WGLEW_I3D_swap_frame_usage = _glewSearchExtension("WGL_I3D_swap_frame_usage", extStart, extEnd);
  if (glewExperimental || WGLEW_I3D_swap_frame_usage|| crippled) WGLEW_I3D_swap_frame_usage= !_glewInit_WGL_I3D_swap_frame_usage(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_I3D_swap_frame_usage */
#ifdef WGL_NV_DX_interop
  WGLEW_NV_DX_interop = _glewSearchExtension("WGL_NV_DX_interop", extStart, extEnd);
  if (glewExperimental || WGLEW_NV_DX_interop|| crippled) WGLEW_NV_DX_interop= !_glewInit_WGL_NV_DX_interop(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_NV_DX_interop */
#ifdef WGL_NV_DX_interop2
  WGLEW_NV_DX_interop2 = _glewSearchExtension("WGL_NV_DX_interop2", extStart, extEnd);
#endif /* WGL_NV_DX_interop2 */
#ifdef WGL_NV_copy_image
  WGLEW_NV_copy_image = _glewSearchExtension("WGL_NV_copy_image", extStart, extEnd);
  if (glewExperimental || WGLEW_NV_copy_image|| crippled) WGLEW_NV_copy_image= !_glewInit_WGL_NV_copy_image(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_NV_copy_image */
#ifdef WGL_NV_delay_before_swap
  WGLEW_NV_delay_before_swap = _glewSearchExtension("WGL_NV_delay_before_swap", extStart, extEnd);
  if (glewExperimental || WGLEW_NV_delay_before_swap|| crippled) WGLEW_NV_delay_before_swap= !_glewInit_WGL_NV_delay_before_swap(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_NV_delay_before_swap */
#ifdef WGL_NV_float_buffer
  WGLEW_NV_float_buffer = _glewSearchExtension("WGL_NV_float_buffer", extStart, extEnd);
#endif /* WGL_NV_float_buffer */
#ifdef WGL_NV_gpu_affinity
  WGLEW_NV_gpu_affinity = _glewSearchExtension("WGL_NV_gpu_affinity", extStart, extEnd);
  if (glewExperimental || WGLEW_NV_gpu_affinity|| crippled) WGLEW_NV_gpu_affinity= !_glewInit_WGL_NV_gpu_affinity(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_NV_gpu_affinity */
#ifdef WGL_NV_multisample_coverage
  WGLEW_NV_multisample_coverage = _glewSearchExtension("WGL_NV_multisample_coverage", extStart, extEnd);
#endif /* WGL_NV_multisample_coverage */
#ifdef WGL_NV_present_video
  WGLEW_NV_present_video = _glewSearchExtension("WGL_NV_present_video", extStart, extEnd);
  if (glewExperimental || WGLEW_NV_present_video|| crippled) WGLEW_NV_present_video= !_glewInit_WGL_NV_present_video(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_NV_present_video */
#ifdef WGL_NV_render_depth_texture
  WGLEW_NV_render_depth_texture = _glewSearchExtension("WGL_NV_render_depth_texture", extStart, extEnd);
#endif /* WGL_NV_render_depth_texture */
#ifdef WGL_NV_render_texture_rectangle
  WGLEW_NV_render_texture_rectangle = _glewSearchExtension("WGL_NV_render_texture_rectangle", extStart, extEnd);
#endif /* WGL_NV_render_texture_rectangle */
#ifdef WGL_NV_swap_group
  WGLEW_NV_swap_group = _glewSearchExtension("WGL_NV_swap_group", extStart, extEnd);
  if (glewExperimental || WGLEW_NV_swap_group|| crippled) WGLEW_NV_swap_group= !_glewInit_WGL_NV_swap_group(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_NV_swap_group */
#ifdef WGL_NV_vertex_array_range
  WGLEW_NV_vertex_array_range = _glewSearchExtension("WGL_NV_vertex_array_range", extStart, extEnd);
  if (glewExperimental || WGLEW_NV_vertex_array_range|| crippled) WGLEW_NV_vertex_array_range= !_glewInit_WGL_NV_vertex_array_range(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_NV_vertex_array_range */
#ifdef WGL_NV_video_capture
  WGLEW_NV_video_capture = _glewSearchExtension("WGL_NV_video_capture", extStart, extEnd);
  if (glewExperimental || WGLEW_NV_video_capture|| crippled) WGLEW_NV_video_capture= !_glewInit_WGL_NV_video_capture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_NV_video_capture */
#ifdef WGL_NV_video_output
  WGLEW_NV_video_output = _glewSearchExtension("WGL_NV_video_output", extStart, extEnd);
  if (glewExperimental || WGLEW_NV_video_output|| crippled) WGLEW_NV_video_output= !_glewInit_WGL_NV_video_output(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_NV_video_output */
#ifdef WGL_OML_sync_control
  WGLEW_OML_sync_control = _glewSearchExtension("WGL_OML_sync_control", extStart, extEnd);
  if (glewExperimental || WGLEW_OML_sync_control|| crippled) WGLEW_OML_sync_control= !_glewInit_WGL_OML_sync_control(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* WGL_OML_sync_control */

  return GLEW_OK;
}

#elif !defined(__ANDROID__) && !defined(__native_client__) && !defined(__HAIKU__) && (!defined(__APPLE__) || defined(GLEW_APPLE_GLX))

PFNGLXGETCURRENTDISPLAYPROC __glewXGetCurrentDisplay = NULL;

PFNGLXCHOOSEFBCONFIGPROC __glewXChooseFBConfig = NULL;
PFNGLXCREATENEWCONTEXTPROC __glewXCreateNewContext = NULL;
PFNGLXCREATEPBUFFERPROC __glewXCreatePbuffer = NULL;
PFNGLXCREATEPIXMAPPROC __glewXCreatePixmap = NULL;
PFNGLXCREATEWINDOWPROC __glewXCreateWindow = NULL;
PFNGLXDESTROYPBUFFERPROC __glewXDestroyPbuffer = NULL;
PFNGLXDESTROYPIXMAPPROC __glewXDestroyPixmap = NULL;
PFNGLXDESTROYWINDOWPROC __glewXDestroyWindow = NULL;
PFNGLXGETCURRENTREADDRAWABLEPROC __glewXGetCurrentReadDrawable = NULL;
PFNGLXGETFBCONFIGATTRIBPROC __glewXGetFBConfigAttrib = NULL;
PFNGLXGETFBCONFIGSPROC __glewXGetFBConfigs = NULL;
PFNGLXGETSELECTEDEVENTPROC __glewXGetSelectedEvent = NULL;
PFNGLXGETVISUALFROMFBCONFIGPROC __glewXGetVisualFromFBConfig = NULL;
PFNGLXMAKECONTEXTCURRENTPROC __glewXMakeContextCurrent = NULL;
PFNGLXQUERYCONTEXTPROC __glewXQueryContext = NULL;
PFNGLXQUERYDRAWABLEPROC __glewXQueryDrawable = NULL;
PFNGLXSELECTEVENTPROC __glewXSelectEvent = NULL;

PFNGLXBLITCONTEXTFRAMEBUFFERAMDPROC __glewXBlitContextFramebufferAMD = NULL;
PFNGLXCREATEASSOCIATEDCONTEXTAMDPROC __glewXCreateAssociatedContextAMD = NULL;
PFNGLXCREATEASSOCIATEDCONTEXTATTRIBSAMDPROC __glewXCreateAssociatedContextAttribsAMD = NULL;
PFNGLXDELETEASSOCIATEDCONTEXTAMDPROC __glewXDeleteAssociatedContextAMD = NULL;
PFNGLXGETCONTEXTGPUIDAMDPROC __glewXGetContextGPUIDAMD = NULL;
PFNGLXGETCURRENTASSOCIATEDCONTEXTAMDPROC __glewXGetCurrentAssociatedContextAMD = NULL;
PFNGLXGETGPUIDSAMDPROC __glewXGetGPUIDsAMD = NULL;
PFNGLXGETGPUINFOAMDPROC __glewXGetGPUInfoAMD = NULL;
PFNGLXMAKEASSOCIATEDCONTEXTCURRENTAMDPROC __glewXMakeAssociatedContextCurrentAMD = NULL;

PFNGLXCREATECONTEXTATTRIBSARBPROC __glewXCreateContextAttribsARB = NULL;

PFNGLXBINDTEXIMAGEATIPROC __glewXBindTexImageATI = NULL;
PFNGLXDRAWABLEATTRIBATIPROC __glewXDrawableAttribATI = NULL;
PFNGLXRELEASETEXIMAGEATIPROC __glewXReleaseTexImageATI = NULL;

PFNGLXFREECONTEXTEXTPROC __glewXFreeContextEXT = NULL;
PFNGLXGETCONTEXTIDEXTPROC __glewXGetContextIDEXT = NULL;
PFNGLXIMPORTCONTEXTEXTPROC __glewXImportContextEXT = NULL;
PFNGLXQUERYCONTEXTINFOEXTPROC __glewXQueryContextInfoEXT = NULL;

PFNGLXSWAPINTERVALEXTPROC __glewXSwapIntervalEXT = NULL;

PFNGLXBINDTEXIMAGEEXTPROC __glewXBindTexImageEXT = NULL;
PFNGLXRELEASETEXIMAGEEXTPROC __glewXReleaseTexImageEXT = NULL;

PFNGLXGETAGPOFFSETMESAPROC __glewXGetAGPOffsetMESA = NULL;

PFNGLXCOPYSUBBUFFERMESAPROC __glewXCopySubBufferMESA = NULL;

PFNGLXCREATEGLXPIXMAPMESAPROC __glewXCreateGLXPixmapMESA = NULL;

PFNGLXQUERYCURRENTRENDERERINTEGERMESAPROC __glewXQueryCurrentRendererIntegerMESA = NULL;
PFNGLXQUERYCURRENTRENDERERSTRINGMESAPROC __glewXQueryCurrentRendererStringMESA = NULL;
PFNGLXQUERYRENDERERINTEGERMESAPROC __glewXQueryRendererIntegerMESA = NULL;
PFNGLXQUERYRENDERERSTRINGMESAPROC __glewXQueryRendererStringMESA = NULL;

PFNGLXRELEASEBUFFERSMESAPROC __glewXReleaseBuffersMESA = NULL;

PFNGLXSET3DFXMODEMESAPROC __glewXSet3DfxModeMESA = NULL;

PFNGLXGETSWAPINTERVALMESAPROC __glewXGetSwapIntervalMESA = NULL;
PFNGLXSWAPINTERVALMESAPROC __glewXSwapIntervalMESA = NULL;

PFNGLXCOPYBUFFERSUBDATANVPROC __glewXCopyBufferSubDataNV = NULL;
PFNGLXNAMEDCOPYBUFFERSUBDATANVPROC __glewXNamedCopyBufferSubDataNV = NULL;

PFNGLXCOPYIMAGESUBDATANVPROC __glewXCopyImageSubDataNV = NULL;

PFNGLXDELAYBEFORESWAPNVPROC __glewXDelayBeforeSwapNV = NULL;

PFNGLXBINDVIDEODEVICENVPROC __glewXBindVideoDeviceNV = NULL;
PFNGLXENUMERATEVIDEODEVICESNVPROC __glewXEnumerateVideoDevicesNV = NULL;

PFNGLXBINDSWAPBARRIERNVPROC __glewXBindSwapBarrierNV = NULL;
PFNGLXJOINSWAPGROUPNVPROC __glewXJoinSwapGroupNV = NULL;
PFNGLXQUERYFRAMECOUNTNVPROC __glewXQueryFrameCountNV = NULL;
PFNGLXQUERYMAXSWAPGROUPSNVPROC __glewXQueryMaxSwapGroupsNV = NULL;
PFNGLXQUERYSWAPGROUPNVPROC __glewXQuerySwapGroupNV = NULL;
PFNGLXRESETFRAMECOUNTNVPROC __glewXResetFrameCountNV = NULL;

PFNGLXALLOCATEMEMORYNVPROC __glewXAllocateMemoryNV = NULL;
PFNGLXFREEMEMORYNVPROC __glewXFreeMemoryNV = NULL;

PFNGLXBINDVIDEOCAPTUREDEVICENVPROC __glewXBindVideoCaptureDeviceNV = NULL;
PFNGLXENUMERATEVIDEOCAPTUREDEVICESNVPROC __glewXEnumerateVideoCaptureDevicesNV = NULL;
PFNGLXLOCKVIDEOCAPTUREDEVICENVPROC __glewXLockVideoCaptureDeviceNV = NULL;
PFNGLXQUERYVIDEOCAPTUREDEVICENVPROC __glewXQueryVideoCaptureDeviceNV = NULL;
PFNGLXRELEASEVIDEOCAPTUREDEVICENVPROC __glewXReleaseVideoCaptureDeviceNV = NULL;

PFNGLXBINDVIDEOIMAGENVPROC __glewXBindVideoImageNV = NULL;
PFNGLXGETVIDEODEVICENVPROC __glewXGetVideoDeviceNV = NULL;
PFNGLXGETVIDEOINFONVPROC __glewXGetVideoInfoNV = NULL;
PFNGLXRELEASEVIDEODEVICENVPROC __glewXReleaseVideoDeviceNV = NULL;
PFNGLXRELEASEVIDEOIMAGENVPROC __glewXReleaseVideoImageNV = NULL;
PFNGLXSENDPBUFFERTOVIDEONVPROC __glewXSendPbufferToVideoNV = NULL;

PFNGLXGETMSCRATEOMLPROC __glewXGetMscRateOML = NULL;
PFNGLXGETSYNCVALUESOMLPROC __glewXGetSyncValuesOML = NULL;
PFNGLXSWAPBUFFERSMSCOMLPROC __glewXSwapBuffersMscOML = NULL;
PFNGLXWAITFORMSCOMLPROC __glewXWaitForMscOML = NULL;
PFNGLXWAITFORSBCOMLPROC __glewXWaitForSbcOML = NULL;

PFNGLXCHOOSEFBCONFIGSGIXPROC __glewXChooseFBConfigSGIX = NULL;
PFNGLXCREATECONTEXTWITHCONFIGSGIXPROC __glewXCreateContextWithConfigSGIX = NULL;
PFNGLXCREATEGLXPIXMAPWITHCONFIGSGIXPROC __glewXCreateGLXPixmapWithConfigSGIX = NULL;
PFNGLXGETFBCONFIGATTRIBSGIXPROC __glewXGetFBConfigAttribSGIX = NULL;
PFNGLXGETFBCONFIGFROMVISUALSGIXPROC __glewXGetFBConfigFromVisualSGIX = NULL;
PFNGLXGETVISUALFROMFBCONFIGSGIXPROC __glewXGetVisualFromFBConfigSGIX = NULL;

PFNGLXBINDHYPERPIPESGIXPROC __glewXBindHyperpipeSGIX = NULL;
PFNGLXDESTROYHYPERPIPECONFIGSGIXPROC __glewXDestroyHyperpipeConfigSGIX = NULL;
PFNGLXHYPERPIPEATTRIBSGIXPROC __glewXHyperpipeAttribSGIX = NULL;
PFNGLXHYPERPIPECONFIGSGIXPROC __glewXHyperpipeConfigSGIX = NULL;
PFNGLXQUERYHYPERPIPEATTRIBSGIXPROC __glewXQueryHyperpipeAttribSGIX = NULL;
PFNGLXQUERYHYPERPIPEBESTATTRIBSGIXPROC __glewXQueryHyperpipeBestAttribSGIX = NULL;
PFNGLXQUERYHYPERPIPECONFIGSGIXPROC __glewXQueryHyperpipeConfigSGIX = NULL;
PFNGLXQUERYHYPERPIPENETWORKSGIXPROC __glewXQueryHyperpipeNetworkSGIX = NULL;

PFNGLXCREATEGLXPBUFFERSGIXPROC __glewXCreateGLXPbufferSGIX = NULL;
PFNGLXDESTROYGLXPBUFFERSGIXPROC __glewXDestroyGLXPbufferSGIX = NULL;
PFNGLXGETSELECTEDEVENTSGIXPROC __glewXGetSelectedEventSGIX = NULL;
PFNGLXQUERYGLXPBUFFERSGIXPROC __glewXQueryGLXPbufferSGIX = NULL;
PFNGLXSELECTEVENTSGIXPROC __glewXSelectEventSGIX = NULL;

PFNGLXBINDSWAPBARRIERSGIXPROC __glewXBindSwapBarrierSGIX = NULL;
PFNGLXQUERYMAXSWAPBARRIERSSGIXPROC __glewXQueryMaxSwapBarriersSGIX = NULL;

PFNGLXJOINSWAPGROUPSGIXPROC __glewXJoinSwapGroupSGIX = NULL;

PFNGLXBINDCHANNELTOWINDOWSGIXPROC __glewXBindChannelToWindowSGIX = NULL;
PFNGLXCHANNELRECTSGIXPROC __glewXChannelRectSGIX = NULL;
PFNGLXCHANNELRECTSYNCSGIXPROC __glewXChannelRectSyncSGIX = NULL;
PFNGLXQUERYCHANNELDELTASSGIXPROC __glewXQueryChannelDeltasSGIX = NULL;
PFNGLXQUERYCHANNELRECTSGIXPROC __glewXQueryChannelRectSGIX = NULL;

PFNGLXCUSHIONSGIPROC __glewXCushionSGI = NULL;

PFNGLXGETCURRENTREADDRAWABLESGIPROC __glewXGetCurrentReadDrawableSGI = NULL;
PFNGLXMAKECURRENTREADSGIPROC __glewXMakeCurrentReadSGI = NULL;

PFNGLXSWAPINTERVALSGIPROC __glewXSwapIntervalSGI = NULL;

PFNGLXGETVIDEOSYNCSGIPROC __glewXGetVideoSyncSGI = NULL;
PFNGLXWAITVIDEOSYNCSGIPROC __glewXWaitVideoSyncSGI = NULL;

PFNGLXGETTRANSPARENTINDEXSUNPROC __glewXGetTransparentIndexSUN = NULL;

PFNGLXGETVIDEORESIZESUNPROC __glewXGetVideoResizeSUN = NULL;
PFNGLXVIDEORESIZESUNPROC __glewXVideoResizeSUN = NULL;

#if !defined(GLEW_MX)

GLboolean __GLXEW_VERSION_1_0 = GL_FALSE;
GLboolean __GLXEW_VERSION_1_1 = GL_FALSE;
GLboolean __GLXEW_VERSION_1_2 = GL_FALSE;
GLboolean __GLXEW_VERSION_1_3 = GL_FALSE;
GLboolean __GLXEW_VERSION_1_4 = GL_FALSE;
GLboolean __GLXEW_3DFX_multisample = GL_FALSE;
GLboolean __GLXEW_AMD_gpu_association = GL_FALSE;
GLboolean __GLXEW_ARB_context_flush_control = GL_FALSE;
GLboolean __GLXEW_ARB_create_context = GL_FALSE;
GLboolean __GLXEW_ARB_create_context_profile = GL_FALSE;
GLboolean __GLXEW_ARB_create_context_robustness = GL_FALSE;
GLboolean __GLXEW_ARB_fbconfig_float = GL_FALSE;
GLboolean __GLXEW_ARB_framebuffer_sRGB = GL_FALSE;
GLboolean __GLXEW_ARB_get_proc_address = GL_FALSE;
GLboolean __GLXEW_ARB_multisample = GL_FALSE;
GLboolean __GLXEW_ARB_robustness_application_isolation = GL_FALSE;
GLboolean __GLXEW_ARB_robustness_share_group_isolation = GL_FALSE;
GLboolean __GLXEW_ARB_vertex_buffer_object = GL_FALSE;
GLboolean __GLXEW_ATI_pixel_format_float = GL_FALSE;
GLboolean __GLXEW_ATI_render_texture = GL_FALSE;
GLboolean __GLXEW_EXT_buffer_age = GL_FALSE;
GLboolean __GLXEW_EXT_create_context_es2_profile = GL_FALSE;
GLboolean __GLXEW_EXT_create_context_es_profile = GL_FALSE;
GLboolean __GLXEW_EXT_fbconfig_packed_float = GL_FALSE;
GLboolean __GLXEW_EXT_framebuffer_sRGB = GL_FALSE;
GLboolean __GLXEW_EXT_import_context = GL_FALSE;
GLboolean __GLXEW_EXT_scene_marker = GL_FALSE;
GLboolean __GLXEW_EXT_stereo_tree = GL_FALSE;
GLboolean __GLXEW_EXT_swap_control = GL_FALSE;
GLboolean __GLXEW_EXT_swap_control_tear = GL_FALSE;
GLboolean __GLXEW_EXT_texture_from_pixmap = GL_FALSE;
GLboolean __GLXEW_EXT_visual_info = GL_FALSE;
GLboolean __GLXEW_EXT_visual_rating = GL_FALSE;
GLboolean __GLXEW_INTEL_swap_event = GL_FALSE;
GLboolean __GLXEW_MESA_agp_offset = GL_FALSE;
GLboolean __GLXEW_MESA_copy_sub_buffer = GL_FALSE;
GLboolean __GLXEW_MESA_pixmap_colormap = GL_FALSE;
GLboolean __GLXEW_MESA_query_renderer = GL_FALSE;
GLboolean __GLXEW_MESA_release_buffers = GL_FALSE;
GLboolean __GLXEW_MESA_set_3dfx_mode = GL_FALSE;
GLboolean __GLXEW_MESA_swap_control = GL_FALSE;
GLboolean __GLXEW_NV_copy_buffer = GL_FALSE;
GLboolean __GLXEW_NV_copy_image = GL_FALSE;
GLboolean __GLXEW_NV_delay_before_swap = GL_FALSE;
GLboolean __GLXEW_NV_float_buffer = GL_FALSE;
GLboolean __GLXEW_NV_multisample_coverage = GL_FALSE;
GLboolean __GLXEW_NV_present_video = GL_FALSE;
GLboolean __GLXEW_NV_swap_group = GL_FALSE;
GLboolean __GLXEW_NV_vertex_array_range = GL_FALSE;
GLboolean __GLXEW_NV_video_capture = GL_FALSE;
GLboolean __GLXEW_NV_video_out = GL_FALSE;
GLboolean __GLXEW_OML_swap_method = GL_FALSE;
GLboolean __GLXEW_OML_sync_control = GL_FALSE;
GLboolean __GLXEW_SGIS_blended_overlay = GL_FALSE;
GLboolean __GLXEW_SGIS_color_range = GL_FALSE;
GLboolean __GLXEW_SGIS_multisample = GL_FALSE;
GLboolean __GLXEW_SGIS_shared_multisample = GL_FALSE;
GLboolean __GLXEW_SGIX_fbconfig = GL_FALSE;
GLboolean __GLXEW_SGIX_hyperpipe = GL_FALSE;
GLboolean __GLXEW_SGIX_pbuffer = GL_FALSE;
GLboolean __GLXEW_SGIX_swap_barrier = GL_FALSE;
GLboolean __GLXEW_SGIX_swap_group = GL_FALSE;
GLboolean __GLXEW_SGIX_video_resize = GL_FALSE;
GLboolean __GLXEW_SGIX_visual_select_group = GL_FALSE;
GLboolean __GLXEW_SGI_cushion = GL_FALSE;
GLboolean __GLXEW_SGI_make_current_read = GL_FALSE;
GLboolean __GLXEW_SGI_swap_control = GL_FALSE;
GLboolean __GLXEW_SGI_video_sync = GL_FALSE;
GLboolean __GLXEW_SUN_get_transparent_index = GL_FALSE;
GLboolean __GLXEW_SUN_video_resize = GL_FALSE;

#endif /* !GLEW_MX */

#ifdef GLX_VERSION_1_2

static GLboolean _glewInit_GLX_VERSION_1_2 (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXGetCurrentDisplay = (PFNGLXGETCURRENTDISPLAYPROC)glewGetProcAddress((const GLubyte*)"glXGetCurrentDisplay")) == NULL) || r;

  return r;
}

#endif /* GLX_VERSION_1_2 */

#ifdef GLX_VERSION_1_3

static GLboolean _glewInit_GLX_VERSION_1_3 (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXChooseFBConfig = (PFNGLXCHOOSEFBCONFIGPROC)glewGetProcAddress((const GLubyte*)"glXChooseFBConfig")) == NULL) || r;
  r = ((glXCreateNewContext = (PFNGLXCREATENEWCONTEXTPROC)glewGetProcAddress((const GLubyte*)"glXCreateNewContext")) == NULL) || r;
  r = ((glXCreatePbuffer = (PFNGLXCREATEPBUFFERPROC)glewGetProcAddress((const GLubyte*)"glXCreatePbuffer")) == NULL) || r;
  r = ((glXCreatePixmap = (PFNGLXCREATEPIXMAPPROC)glewGetProcAddress((const GLubyte*)"glXCreatePixmap")) == NULL) || r;
  r = ((glXCreateWindow = (PFNGLXCREATEWINDOWPROC)glewGetProcAddress((const GLubyte*)"glXCreateWindow")) == NULL) || r;
  r = ((glXDestroyPbuffer = (PFNGLXDESTROYPBUFFERPROC)glewGetProcAddress((const GLubyte*)"glXDestroyPbuffer")) == NULL) || r;
  r = ((glXDestroyPixmap = (PFNGLXDESTROYPIXMAPPROC)glewGetProcAddress((const GLubyte*)"glXDestroyPixmap")) == NULL) || r;
  r = ((glXDestroyWindow = (PFNGLXDESTROYWINDOWPROC)glewGetProcAddress((const GLubyte*)"glXDestroyWindow")) == NULL) || r;
  r = ((glXGetCurrentReadDrawable = (PFNGLXGETCURRENTREADDRAWABLEPROC)glewGetProcAddress((const GLubyte*)"glXGetCurrentReadDrawable")) == NULL) || r;
  r = ((glXGetFBConfigAttrib = (PFNGLXGETFBCONFIGATTRIBPROC)glewGetProcAddress((const GLubyte*)"glXGetFBConfigAttrib")) == NULL) || r;
  r = ((glXGetFBConfigs = (PFNGLXGETFBCONFIGSPROC)glewGetProcAddress((const GLubyte*)"glXGetFBConfigs")) == NULL) || r;
  r = ((glXGetSelectedEvent = (PFNGLXGETSELECTEDEVENTPROC)glewGetProcAddress((const GLubyte*)"glXGetSelectedEvent")) == NULL) || r;
  r = ((glXGetVisualFromFBConfig = (PFNGLXGETVISUALFROMFBCONFIGPROC)glewGetProcAddress((const GLubyte*)"glXGetVisualFromFBConfig")) == NULL) || r;
  r = ((glXMakeContextCurrent = (PFNGLXMAKECONTEXTCURRENTPROC)glewGetProcAddress((const GLubyte*)"glXMakeContextCurrent")) == NULL) || r;
  r = ((glXQueryContext = (PFNGLXQUERYCONTEXTPROC)glewGetProcAddress((const GLubyte*)"glXQueryContext")) == NULL) || r;
  r = ((glXQueryDrawable = (PFNGLXQUERYDRAWABLEPROC)glewGetProcAddress((const GLubyte*)"glXQueryDrawable")) == NULL) || r;
  r = ((glXSelectEvent = (PFNGLXSELECTEVENTPROC)glewGetProcAddress((const GLubyte*)"glXSelectEvent")) == NULL) || r;

  return r;
}

#endif /* GLX_VERSION_1_3 */

#ifdef GLX_AMD_gpu_association

static GLboolean _glewInit_GLX_AMD_gpu_association (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXBlitContextFramebufferAMD = (PFNGLXBLITCONTEXTFRAMEBUFFERAMDPROC)glewGetProcAddress((const GLubyte*)"glXBlitContextFramebufferAMD")) == NULL) || r;
  r = ((glXCreateAssociatedContextAMD = (PFNGLXCREATEASSOCIATEDCONTEXTAMDPROC)glewGetProcAddress((const GLubyte*)"glXCreateAssociatedContextAMD")) == NULL) || r;
  r = ((glXCreateAssociatedContextAttribsAMD = (PFNGLXCREATEASSOCIATEDCONTEXTATTRIBSAMDPROC)glewGetProcAddress((const GLubyte*)"glXCreateAssociatedContextAttribsAMD")) == NULL) || r;
  r = ((glXDeleteAssociatedContextAMD = (PFNGLXDELETEASSOCIATEDCONTEXTAMDPROC)glewGetProcAddress((const GLubyte*)"glXDeleteAssociatedContextAMD")) == NULL) || r;
  r = ((glXGetContextGPUIDAMD = (PFNGLXGETCONTEXTGPUIDAMDPROC)glewGetProcAddress((const GLubyte*)"glXGetContextGPUIDAMD")) == NULL) || r;
  r = ((glXGetCurrentAssociatedContextAMD = (PFNGLXGETCURRENTASSOCIATEDCONTEXTAMDPROC)glewGetProcAddress((const GLubyte*)"glXGetCurrentAssociatedContextAMD")) == NULL) || r;
  r = ((glXGetGPUIDsAMD = (PFNGLXGETGPUIDSAMDPROC)glewGetProcAddress((const GLubyte*)"glXGetGPUIDsAMD")) == NULL) || r;
  r = ((glXGetGPUInfoAMD = (PFNGLXGETGPUINFOAMDPROC)glewGetProcAddress((const GLubyte*)"glXGetGPUInfoAMD")) == NULL) || r;
  r = ((glXMakeAssociatedContextCurrentAMD = (PFNGLXMAKEASSOCIATEDCONTEXTCURRENTAMDPROC)glewGetProcAddress((const GLubyte*)"glXMakeAssociatedContextCurrentAMD")) == NULL) || r;

  return r;
}

#endif /* GLX_AMD_gpu_association */

#ifdef GLX_ARB_create_context

static GLboolean _glewInit_GLX_ARB_create_context (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXCreateContextAttribsARB = (PFNGLXCREATECONTEXTATTRIBSARBPROC)glewGetProcAddress((const GLubyte*)"glXCreateContextAttribsARB")) == NULL) || r;

  return r;
}

#endif /* GLX_ARB_create_context */

#ifdef GLX_ATI_render_texture

static GLboolean _glewInit_GLX_ATI_render_texture (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXBindTexImageATI = (PFNGLXBINDTEXIMAGEATIPROC)glewGetProcAddress((const GLubyte*)"glXBindTexImageATI")) == NULL) || r;
  r = ((glXDrawableAttribATI = (PFNGLXDRAWABLEATTRIBATIPROC)glewGetProcAddress((const GLubyte*)"glXDrawableAttribATI")) == NULL) || r;
  r = ((glXReleaseTexImageATI = (PFNGLXRELEASETEXIMAGEATIPROC)glewGetProcAddress((const GLubyte*)"glXReleaseTexImageATI")) == NULL) || r;

  return r;
}

#endif /* GLX_ATI_render_texture */

#ifdef GLX_EXT_import_context

static GLboolean _glewInit_GLX_EXT_import_context (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXFreeContextEXT = (PFNGLXFREECONTEXTEXTPROC)glewGetProcAddress((const GLubyte*)"glXFreeContextEXT")) == NULL) || r;
  r = ((glXGetContextIDEXT = (PFNGLXGETCONTEXTIDEXTPROC)glewGetProcAddress((const GLubyte*)"glXGetContextIDEXT")) == NULL) || r;
  r = ((glXImportContextEXT = (PFNGLXIMPORTCONTEXTEXTPROC)glewGetProcAddress((const GLubyte*)"glXImportContextEXT")) == NULL) || r;
  r = ((glXQueryContextInfoEXT = (PFNGLXQUERYCONTEXTINFOEXTPROC)glewGetProcAddress((const GLubyte*)"glXQueryContextInfoEXT")) == NULL) || r;

  return r;
}

#endif /* GLX_EXT_import_context */

#ifdef GLX_EXT_swap_control

static GLboolean _glewInit_GLX_EXT_swap_control (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)glewGetProcAddress((const GLubyte*)"glXSwapIntervalEXT")) == NULL) || r;

  return r;
}

#endif /* GLX_EXT_swap_control */

#ifdef GLX_EXT_texture_from_pixmap

static GLboolean _glewInit_GLX_EXT_texture_from_pixmap (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXBindTexImageEXT = (PFNGLXBINDTEXIMAGEEXTPROC)glewGetProcAddress((const GLubyte*)"glXBindTexImageEXT")) == NULL) || r;
  r = ((glXReleaseTexImageEXT = (PFNGLXRELEASETEXIMAGEEXTPROC)glewGetProcAddress((const GLubyte*)"glXReleaseTexImageEXT")) == NULL) || r;

  return r;
}

#endif /* GLX_EXT_texture_from_pixmap */

#ifdef GLX_MESA_agp_offset

static GLboolean _glewInit_GLX_MESA_agp_offset (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXGetAGPOffsetMESA = (PFNGLXGETAGPOFFSETMESAPROC)glewGetProcAddress((const GLubyte*)"glXGetAGPOffsetMESA")) == NULL) || r;

  return r;
}

#endif /* GLX_MESA_agp_offset */

#ifdef GLX_MESA_copy_sub_buffer

static GLboolean _glewInit_GLX_MESA_copy_sub_buffer (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXCopySubBufferMESA = (PFNGLXCOPYSUBBUFFERMESAPROC)glewGetProcAddress((const GLubyte*)"glXCopySubBufferMESA")) == NULL) || r;

  return r;
}

#endif /* GLX_MESA_copy_sub_buffer */

#ifdef GLX_MESA_pixmap_colormap

static GLboolean _glewInit_GLX_MESA_pixmap_colormap (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXCreateGLXPixmapMESA = (PFNGLXCREATEGLXPIXMAPMESAPROC)glewGetProcAddress((const GLubyte*)"glXCreateGLXPixmapMESA")) == NULL) || r;

  return r;
}

#endif /* GLX_MESA_pixmap_colormap */

#ifdef GLX_MESA_query_renderer

static GLboolean _glewInit_GLX_MESA_query_renderer (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXQueryCurrentRendererIntegerMESA = (PFNGLXQUERYCURRENTRENDERERINTEGERMESAPROC)glewGetProcAddress((const GLubyte*)"glXQueryCurrentRendererIntegerMESA")) == NULL) || r;
  r = ((glXQueryCurrentRendererStringMESA = (PFNGLXQUERYCURRENTRENDERERSTRINGMESAPROC)glewGetProcAddress((const GLubyte*)"glXQueryCurrentRendererStringMESA")) == NULL) || r;
  r = ((glXQueryRendererIntegerMESA = (PFNGLXQUERYRENDERERINTEGERMESAPROC)glewGetProcAddress((const GLubyte*)"glXQueryRendererIntegerMESA")) == NULL) || r;
  r = ((glXQueryRendererStringMESA = (PFNGLXQUERYRENDERERSTRINGMESAPROC)glewGetProcAddress((const GLubyte*)"glXQueryRendererStringMESA")) == NULL) || r;

  return r;
}

#endif /* GLX_MESA_query_renderer */

#ifdef GLX_MESA_release_buffers

static GLboolean _glewInit_GLX_MESA_release_buffers (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXReleaseBuffersMESA = (PFNGLXRELEASEBUFFERSMESAPROC)glewGetProcAddress((const GLubyte*)"glXReleaseBuffersMESA")) == NULL) || r;

  return r;
}

#endif /* GLX_MESA_release_buffers */

#ifdef GLX_MESA_set_3dfx_mode

static GLboolean _glewInit_GLX_MESA_set_3dfx_mode (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXSet3DfxModeMESA = (PFNGLXSET3DFXMODEMESAPROC)glewGetProcAddress((const GLubyte*)"glXSet3DfxModeMESA")) == NULL) || r;

  return r;
}

#endif /* GLX_MESA_set_3dfx_mode */

#ifdef GLX_MESA_swap_control

static GLboolean _glewInit_GLX_MESA_swap_control (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXGetSwapIntervalMESA = (PFNGLXGETSWAPINTERVALMESAPROC)glewGetProcAddress((const GLubyte*)"glXGetSwapIntervalMESA")) == NULL) || r;
  r = ((glXSwapIntervalMESA = (PFNGLXSWAPINTERVALMESAPROC)glewGetProcAddress((const GLubyte*)"glXSwapIntervalMESA")) == NULL) || r;

  return r;
}

#endif /* GLX_MESA_swap_control */

#ifdef GLX_NV_copy_buffer

static GLboolean _glewInit_GLX_NV_copy_buffer (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXCopyBufferSubDataNV = (PFNGLXCOPYBUFFERSUBDATANVPROC)glewGetProcAddress((const GLubyte*)"glXCopyBufferSubDataNV")) == NULL) || r;
  r = ((glXNamedCopyBufferSubDataNV = (PFNGLXNAMEDCOPYBUFFERSUBDATANVPROC)glewGetProcAddress((const GLubyte*)"glXNamedCopyBufferSubDataNV")) == NULL) || r;

  return r;
}

#endif /* GLX_NV_copy_buffer */

#ifdef GLX_NV_copy_image

static GLboolean _glewInit_GLX_NV_copy_image (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXCopyImageSubDataNV = (PFNGLXCOPYIMAGESUBDATANVPROC)glewGetProcAddress((const GLubyte*)"glXCopyImageSubDataNV")) == NULL) || r;

  return r;
}

#endif /* GLX_NV_copy_image */

#ifdef GLX_NV_delay_before_swap

static GLboolean _glewInit_GLX_NV_delay_before_swap (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXDelayBeforeSwapNV = (PFNGLXDELAYBEFORESWAPNVPROC)glewGetProcAddress((const GLubyte*)"glXDelayBeforeSwapNV")) == NULL) || r;

  return r;
}

#endif /* GLX_NV_delay_before_swap */

#ifdef GLX_NV_present_video

static GLboolean _glewInit_GLX_NV_present_video (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXBindVideoDeviceNV = (PFNGLXBINDVIDEODEVICENVPROC)glewGetProcAddress((const GLubyte*)"glXBindVideoDeviceNV")) == NULL) || r;
  r = ((glXEnumerateVideoDevicesNV = (PFNGLXENUMERATEVIDEODEVICESNVPROC)glewGetProcAddress((const GLubyte*)"glXEnumerateVideoDevicesNV")) == NULL) || r;

  return r;
}

#endif /* GLX_NV_present_video */

#ifdef GLX_NV_swap_group

static GLboolean _glewInit_GLX_NV_swap_group (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXBindSwapBarrierNV = (PFNGLXBINDSWAPBARRIERNVPROC)glewGetProcAddress((const GLubyte*)"glXBindSwapBarrierNV")) == NULL) || r;
  r = ((glXJoinSwapGroupNV = (PFNGLXJOINSWAPGROUPNVPROC)glewGetProcAddress((const GLubyte*)"glXJoinSwapGroupNV")) == NULL) || r;
  r = ((glXQueryFrameCountNV = (PFNGLXQUERYFRAMECOUNTNVPROC)glewGetProcAddress((const GLubyte*)"glXQueryFrameCountNV")) == NULL) || r;
  r = ((glXQueryMaxSwapGroupsNV = (PFNGLXQUERYMAXSWAPGROUPSNVPROC)glewGetProcAddress((const GLubyte*)"glXQueryMaxSwapGroupsNV")) == NULL) || r;
  r = ((glXQuerySwapGroupNV = (PFNGLXQUERYSWAPGROUPNVPROC)glewGetProcAddress((const GLubyte*)"glXQuerySwapGroupNV")) == NULL) || r;
  r = ((glXResetFrameCountNV = (PFNGLXRESETFRAMECOUNTNVPROC)glewGetProcAddress((const GLubyte*)"glXResetFrameCountNV")) == NULL) || r;

  return r;
}

#endif /* GLX_NV_swap_group */

#ifdef GLX_NV_vertex_array_range

static GLboolean _glewInit_GLX_NV_vertex_array_range (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXAllocateMemoryNV = (PFNGLXALLOCATEMEMORYNVPROC)glewGetProcAddress((const GLubyte*)"glXAllocateMemoryNV")) == NULL) || r;
  r = ((glXFreeMemoryNV = (PFNGLXFREEMEMORYNVPROC)glewGetProcAddress((const GLubyte*)"glXFreeMemoryNV")) == NULL) || r;

  return r;
}

#endif /* GLX_NV_vertex_array_range */

#ifdef GLX_NV_video_capture

static GLboolean _glewInit_GLX_NV_video_capture (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXBindVideoCaptureDeviceNV = (PFNGLXBINDVIDEOCAPTUREDEVICENVPROC)glewGetProcAddress((const GLubyte*)"glXBindVideoCaptureDeviceNV")) == NULL) || r;
  r = ((glXEnumerateVideoCaptureDevicesNV = (PFNGLXENUMERATEVIDEOCAPTUREDEVICESNVPROC)glewGetProcAddress((const GLubyte*)"glXEnumerateVideoCaptureDevicesNV")) == NULL) || r;
  r = ((glXLockVideoCaptureDeviceNV = (PFNGLXLOCKVIDEOCAPTUREDEVICENVPROC)glewGetProcAddress((const GLubyte*)"glXLockVideoCaptureDeviceNV")) == NULL) || r;
  r = ((glXQueryVideoCaptureDeviceNV = (PFNGLXQUERYVIDEOCAPTUREDEVICENVPROC)glewGetProcAddress((const GLubyte*)"glXQueryVideoCaptureDeviceNV")) == NULL) || r;
  r = ((glXReleaseVideoCaptureDeviceNV = (PFNGLXRELEASEVIDEOCAPTUREDEVICENVPROC)glewGetProcAddress((const GLubyte*)"glXReleaseVideoCaptureDeviceNV")) == NULL) || r;

  return r;
}

#endif /* GLX_NV_video_capture */

#ifdef GLX_NV_video_out

static GLboolean _glewInit_GLX_NV_video_out (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXBindVideoImageNV = (PFNGLXBINDVIDEOIMAGENVPROC)glewGetProcAddress((const GLubyte*)"glXBindVideoImageNV")) == NULL) || r;
  r = ((glXGetVideoDeviceNV = (PFNGLXGETVIDEODEVICENVPROC)glewGetProcAddress((const GLubyte*)"glXGetVideoDeviceNV")) == NULL) || r;
  r = ((glXGetVideoInfoNV = (PFNGLXGETVIDEOINFONVPROC)glewGetProcAddress((const GLubyte*)"glXGetVideoInfoNV")) == NULL) || r;
  r = ((glXReleaseVideoDeviceNV = (PFNGLXRELEASEVIDEODEVICENVPROC)glewGetProcAddress((const GLubyte*)"glXReleaseVideoDeviceNV")) == NULL) || r;
  r = ((glXReleaseVideoImageNV = (PFNGLXRELEASEVIDEOIMAGENVPROC)glewGetProcAddress((const GLubyte*)"glXReleaseVideoImageNV")) == NULL) || r;
  r = ((glXSendPbufferToVideoNV = (PFNGLXSENDPBUFFERTOVIDEONVPROC)glewGetProcAddress((const GLubyte*)"glXSendPbufferToVideoNV")) == NULL) || r;

  return r;
}

#endif /* GLX_NV_video_out */

#ifdef GLX_OML_sync_control

static GLboolean _glewInit_GLX_OML_sync_control (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXGetMscRateOML = (PFNGLXGETMSCRATEOMLPROC)glewGetProcAddress((const GLubyte*)"glXGetMscRateOML")) == NULL) || r;
  r = ((glXGetSyncValuesOML = (PFNGLXGETSYNCVALUESOMLPROC)glewGetProcAddress((const GLubyte*)"glXGetSyncValuesOML")) == NULL) || r;
  r = ((glXSwapBuffersMscOML = (PFNGLXSWAPBUFFERSMSCOMLPROC)glewGetProcAddress((const GLubyte*)"glXSwapBuffersMscOML")) == NULL) || r;
  r = ((glXWaitForMscOML = (PFNGLXWAITFORMSCOMLPROC)glewGetProcAddress((const GLubyte*)"glXWaitForMscOML")) == NULL) || r;
  r = ((glXWaitForSbcOML = (PFNGLXWAITFORSBCOMLPROC)glewGetProcAddress((const GLubyte*)"glXWaitForSbcOML")) == NULL) || r;

  return r;
}

#endif /* GLX_OML_sync_control */

#ifdef GLX_SGIX_fbconfig

static GLboolean _glewInit_GLX_SGIX_fbconfig (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXChooseFBConfigSGIX = (PFNGLXCHOOSEFBCONFIGSGIXPROC)glewGetProcAddress((const GLubyte*)"glXChooseFBConfigSGIX")) == NULL) || r;
  r = ((glXCreateContextWithConfigSGIX = (PFNGLXCREATECONTEXTWITHCONFIGSGIXPROC)glewGetProcAddress((const GLubyte*)"glXCreateContextWithConfigSGIX")) == NULL) || r;
  r = ((glXCreateGLXPixmapWithConfigSGIX = (PFNGLXCREATEGLXPIXMAPWITHCONFIGSGIXPROC)glewGetProcAddress((const GLubyte*)"glXCreateGLXPixmapWithConfigSGIX")) == NULL) || r;
  r = ((glXGetFBConfigAttribSGIX = (PFNGLXGETFBCONFIGATTRIBSGIXPROC)glewGetProcAddress((const GLubyte*)"glXGetFBConfigAttribSGIX")) == NULL) || r;
  r = ((glXGetFBConfigFromVisualSGIX = (PFNGLXGETFBCONFIGFROMVISUALSGIXPROC)glewGetProcAddress((const GLubyte*)"glXGetFBConfigFromVisualSGIX")) == NULL) || r;
  r = ((glXGetVisualFromFBConfigSGIX = (PFNGLXGETVISUALFROMFBCONFIGSGIXPROC)glewGetProcAddress((const GLubyte*)"glXGetVisualFromFBConfigSGIX")) == NULL) || r;

  return r;
}

#endif /* GLX_SGIX_fbconfig */

#ifdef GLX_SGIX_hyperpipe

static GLboolean _glewInit_GLX_SGIX_hyperpipe (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXBindHyperpipeSGIX = (PFNGLXBINDHYPERPIPESGIXPROC)glewGetProcAddress((const GLubyte*)"glXBindHyperpipeSGIX")) == NULL) || r;
  r = ((glXDestroyHyperpipeConfigSGIX = (PFNGLXDESTROYHYPERPIPECONFIGSGIXPROC)glewGetProcAddress((const GLubyte*)"glXDestroyHyperpipeConfigSGIX")) == NULL) || r;
  r = ((glXHyperpipeAttribSGIX = (PFNGLXHYPERPIPEATTRIBSGIXPROC)glewGetProcAddress((const GLubyte*)"glXHyperpipeAttribSGIX")) == NULL) || r;
  r = ((glXHyperpipeConfigSGIX = (PFNGLXHYPERPIPECONFIGSGIXPROC)glewGetProcAddress((const GLubyte*)"glXHyperpipeConfigSGIX")) == NULL) || r;
  r = ((glXQueryHyperpipeAttribSGIX = (PFNGLXQUERYHYPERPIPEATTRIBSGIXPROC)glewGetProcAddress((const GLubyte*)"glXQueryHyperpipeAttribSGIX")) == NULL) || r;
  r = ((glXQueryHyperpipeBestAttribSGIX = (PFNGLXQUERYHYPERPIPEBESTATTRIBSGIXPROC)glewGetProcAddress((const GLubyte*)"glXQueryHyperpipeBestAttribSGIX")) == NULL) || r;
  r = ((glXQueryHyperpipeConfigSGIX = (PFNGLXQUERYHYPERPIPECONFIGSGIXPROC)glewGetProcAddress((const GLubyte*)"glXQueryHyperpipeConfigSGIX")) == NULL) || r;
  r = ((glXQueryHyperpipeNetworkSGIX = (PFNGLXQUERYHYPERPIPENETWORKSGIXPROC)glewGetProcAddress((const GLubyte*)"glXQueryHyperpipeNetworkSGIX")) == NULL) || r;

  return r;
}

#endif /* GLX_SGIX_hyperpipe */

#ifdef GLX_SGIX_pbuffer

static GLboolean _glewInit_GLX_SGIX_pbuffer (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXCreateGLXPbufferSGIX = (PFNGLXCREATEGLXPBUFFERSGIXPROC)glewGetProcAddress((const GLubyte*)"glXCreateGLXPbufferSGIX")) == NULL) || r;
  r = ((glXDestroyGLXPbufferSGIX = (PFNGLXDESTROYGLXPBUFFERSGIXPROC)glewGetProcAddress((const GLubyte*)"glXDestroyGLXPbufferSGIX")) == NULL) || r;
  r = ((glXGetSelectedEventSGIX = (PFNGLXGETSELECTEDEVENTSGIXPROC)glewGetProcAddress((const GLubyte*)"glXGetSelectedEventSGIX")) == NULL) || r;
  r = ((glXQueryGLXPbufferSGIX = (PFNGLXQUERYGLXPBUFFERSGIXPROC)glewGetProcAddress((const GLubyte*)"glXQueryGLXPbufferSGIX")) == NULL) || r;
  r = ((glXSelectEventSGIX = (PFNGLXSELECTEVENTSGIXPROC)glewGetProcAddress((const GLubyte*)"glXSelectEventSGIX")) == NULL) || r;

  return r;
}

#endif /* GLX_SGIX_pbuffer */

#ifdef GLX_SGIX_swap_barrier

static GLboolean _glewInit_GLX_SGIX_swap_barrier (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXBindSwapBarrierSGIX = (PFNGLXBINDSWAPBARRIERSGIXPROC)glewGetProcAddress((const GLubyte*)"glXBindSwapBarrierSGIX")) == NULL) || r;
  r = ((glXQueryMaxSwapBarriersSGIX = (PFNGLXQUERYMAXSWAPBARRIERSSGIXPROC)glewGetProcAddress((const GLubyte*)"glXQueryMaxSwapBarriersSGIX")) == NULL) || r;

  return r;
}

#endif /* GLX_SGIX_swap_barrier */

#ifdef GLX_SGIX_swap_group

static GLboolean _glewInit_GLX_SGIX_swap_group (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXJoinSwapGroupSGIX = (PFNGLXJOINSWAPGROUPSGIXPROC)glewGetProcAddress((const GLubyte*)"glXJoinSwapGroupSGIX")) == NULL) || r;

  return r;
}

#endif /* GLX_SGIX_swap_group */

#ifdef GLX_SGIX_video_resize

static GLboolean _glewInit_GLX_SGIX_video_resize (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXBindChannelToWindowSGIX = (PFNGLXBINDCHANNELTOWINDOWSGIXPROC)glewGetProcAddress((const GLubyte*)"glXBindChannelToWindowSGIX")) == NULL) || r;
  r = ((glXChannelRectSGIX = (PFNGLXCHANNELRECTSGIXPROC)glewGetProcAddress((const GLubyte*)"glXChannelRectSGIX")) == NULL) || r;
  r = ((glXChannelRectSyncSGIX = (PFNGLXCHANNELRECTSYNCSGIXPROC)glewGetProcAddress((const GLubyte*)"glXChannelRectSyncSGIX")) == NULL) || r;
  r = ((glXQueryChannelDeltasSGIX = (PFNGLXQUERYCHANNELDELTASSGIXPROC)glewGetProcAddress((const GLubyte*)"glXQueryChannelDeltasSGIX")) == NULL) || r;
  r = ((glXQueryChannelRectSGIX = (PFNGLXQUERYCHANNELRECTSGIXPROC)glewGetProcAddress((const GLubyte*)"glXQueryChannelRectSGIX")) == NULL) || r;

  return r;
}

#endif /* GLX_SGIX_video_resize */

#ifdef GLX_SGI_cushion

static GLboolean _glewInit_GLX_SGI_cushion (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXCushionSGI = (PFNGLXCUSHIONSGIPROC)glewGetProcAddress((const GLubyte*)"glXCushionSGI")) == NULL) || r;

  return r;
}

#endif /* GLX_SGI_cushion */

#ifdef GLX_SGI_make_current_read

static GLboolean _glewInit_GLX_SGI_make_current_read (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXGetCurrentReadDrawableSGI = (PFNGLXGETCURRENTREADDRAWABLESGIPROC)glewGetProcAddress((const GLubyte*)"glXGetCurrentReadDrawableSGI")) == NULL) || r;
  r = ((glXMakeCurrentReadSGI = (PFNGLXMAKECURRENTREADSGIPROC)glewGetProcAddress((const GLubyte*)"glXMakeCurrentReadSGI")) == NULL) || r;

  return r;
}

#endif /* GLX_SGI_make_current_read */

#ifdef GLX_SGI_swap_control

static GLboolean _glewInit_GLX_SGI_swap_control (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXSwapIntervalSGI = (PFNGLXSWAPINTERVALSGIPROC)glewGetProcAddress((const GLubyte*)"glXSwapIntervalSGI")) == NULL) || r;

  return r;
}

#endif /* GLX_SGI_swap_control */

#ifdef GLX_SGI_video_sync

static GLboolean _glewInit_GLX_SGI_video_sync (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXGetVideoSyncSGI = (PFNGLXGETVIDEOSYNCSGIPROC)glewGetProcAddress((const GLubyte*)"glXGetVideoSyncSGI")) == NULL) || r;
  r = ((glXWaitVideoSyncSGI = (PFNGLXWAITVIDEOSYNCSGIPROC)glewGetProcAddress((const GLubyte*)"glXWaitVideoSyncSGI")) == NULL) || r;

  return r;
}

#endif /* GLX_SGI_video_sync */

#ifdef GLX_SUN_get_transparent_index

static GLboolean _glewInit_GLX_SUN_get_transparent_index (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXGetTransparentIndexSUN = (PFNGLXGETTRANSPARENTINDEXSUNPROC)glewGetProcAddress((const GLubyte*)"glXGetTransparentIndexSUN")) == NULL) || r;

  return r;
}

#endif /* GLX_SUN_get_transparent_index */

#ifdef GLX_SUN_video_resize

static GLboolean _glewInit_GLX_SUN_video_resize (GLXEW_CONTEXT_ARG_DEF_INIT)
{
  GLboolean r = GL_FALSE;

  r = ((glXGetVideoResizeSUN = (PFNGLXGETVIDEORESIZESUNPROC)glewGetProcAddress((const GLubyte*)"glXGetVideoResizeSUN")) == NULL) || r;
  r = ((glXVideoResizeSUN = (PFNGLXVIDEORESIZESUNPROC)glewGetProcAddress((const GLubyte*)"glXVideoResizeSUN")) == NULL) || r;

  return r;
}

#endif /* GLX_SUN_video_resize */

/* ------------------------------------------------------------------------ */

GLboolean glxewGetExtension (const char* name)
{    
  const GLubyte* start;
  const GLubyte* end;

  if (glXGetCurrentDisplay == NULL) return GL_FALSE;
  start = (const GLubyte*)glXGetClientString(glXGetCurrentDisplay(), GLX_EXTENSIONS);
  if (0 == start) return GL_FALSE;
  end = start + _glewStrLen(start);
  return _glewSearchExtension(name, start, end);
}

#ifdef GLEW_MX
GLenum glxewContextInit (GLXEW_CONTEXT_ARG_DEF_LIST)
#else
GLenum glxewInit (GLXEW_CONTEXT_ARG_DEF_LIST)
#endif
{
  int major, minor;
  const GLubyte* extStart;
  const GLubyte* extEnd;
  /* initialize core GLX 1.2 */
  if (_glewInit_GLX_VERSION_1_2(GLEW_CONTEXT_ARG_VAR_INIT)) return GLEW_ERROR_GLX_VERSION_11_ONLY;
  /* initialize flags */
  GLXEW_VERSION_1_0 = GL_TRUE;
  GLXEW_VERSION_1_1 = GL_TRUE;
  GLXEW_VERSION_1_2 = GL_TRUE;
  GLXEW_VERSION_1_3 = GL_TRUE;
  GLXEW_VERSION_1_4 = GL_TRUE;
  /* query GLX version */
  glXQueryVersion(glXGetCurrentDisplay(), &major, &minor);
  if (major == 1 && minor <= 3)
  {
    switch (minor)
    {
      case 3:
      GLXEW_VERSION_1_4 = GL_FALSE;
      break;
      case 2:
      GLXEW_VERSION_1_4 = GL_FALSE;
      GLXEW_VERSION_1_3 = GL_FALSE;
      break;
      default:
      return GLEW_ERROR_GLX_VERSION_11_ONLY;
      break;
    }
  }
  /* query GLX extension string */
  extStart = 0;
  if (glXGetCurrentDisplay != NULL)
    extStart = (const GLubyte*)glXGetClientString(glXGetCurrentDisplay(), GLX_EXTENSIONS);
  if (extStart == 0)
    extStart = (const GLubyte *)"";
  extEnd = extStart + _glewStrLen(extStart);
  /* initialize extensions */
#ifdef GLX_VERSION_1_3
  if (glewExperimental || GLXEW_VERSION_1_3) GLXEW_VERSION_1_3 = !_glewInit_GLX_VERSION_1_3(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_VERSION_1_3 */
#ifdef GLX_3DFX_multisample
  GLXEW_3DFX_multisample = _glewSearchExtension("GLX_3DFX_multisample", extStart, extEnd);
#endif /* GLX_3DFX_multisample */
#ifdef GLX_AMD_gpu_association
  GLXEW_AMD_gpu_association = _glewSearchExtension("GLX_AMD_gpu_association", extStart, extEnd);
  if (glewExperimental || GLXEW_AMD_gpu_association) GLXEW_AMD_gpu_association = !_glewInit_GLX_AMD_gpu_association(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_AMD_gpu_association */
#ifdef GLX_ARB_context_flush_control
  GLXEW_ARB_context_flush_control = _glewSearchExtension("GLX_ARB_context_flush_control", extStart, extEnd);
#endif /* GLX_ARB_context_flush_control */
#ifdef GLX_ARB_create_context
  GLXEW_ARB_create_context = _glewSearchExtension("GLX_ARB_create_context", extStart, extEnd);
  if (glewExperimental || GLXEW_ARB_create_context) GLXEW_ARB_create_context = !_glewInit_GLX_ARB_create_context(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_ARB_create_context */
#ifdef GLX_ARB_create_context_profile
  GLXEW_ARB_create_context_profile = _glewSearchExtension("GLX_ARB_create_context_profile", extStart, extEnd);
#endif /* GLX_ARB_create_context_profile */
#ifdef GLX_ARB_create_context_robustness
  GLXEW_ARB_create_context_robustness = _glewSearchExtension("GLX_ARB_create_context_robustness", extStart, extEnd);
#endif /* GLX_ARB_create_context_robustness */
#ifdef GLX_ARB_fbconfig_float
  GLXEW_ARB_fbconfig_float = _glewSearchExtension("GLX_ARB_fbconfig_float", extStart, extEnd);
#endif /* GLX_ARB_fbconfig_float */
#ifdef GLX_ARB_framebuffer_sRGB
  GLXEW_ARB_framebuffer_sRGB = _glewSearchExtension("GLX_ARB_framebuffer_sRGB", extStart, extEnd);
#endif /* GLX_ARB_framebuffer_sRGB */
#ifdef GLX_ARB_get_proc_address
  GLXEW_ARB_get_proc_address = _glewSearchExtension("GLX_ARB_get_proc_address", extStart, extEnd);
#endif /* GLX_ARB_get_proc_address */
#ifdef GLX_ARB_multisample
  GLXEW_ARB_multisample = _glewSearchExtension("GLX_ARB_multisample", extStart, extEnd);
#endif /* GLX_ARB_multisample */
#ifdef GLX_ARB_robustness_application_isolation
  GLXEW_ARB_robustness_application_isolation = _glewSearchExtension("GLX_ARB_robustness_application_isolation", extStart, extEnd);
#endif /* GLX_ARB_robustness_application_isolation */
#ifdef GLX_ARB_robustness_share_group_isolation
  GLXEW_ARB_robustness_share_group_isolation = _glewSearchExtension("GLX_ARB_robustness_share_group_isolation", extStart, extEnd);
#endif /* GLX_ARB_robustness_share_group_isolation */
#ifdef GLX_ARB_vertex_buffer_object
  GLXEW_ARB_vertex_buffer_object = _glewSearchExtension("GLX_ARB_vertex_buffer_object", extStart, extEnd);
#endif /* GLX_ARB_vertex_buffer_object */
#ifdef GLX_ATI_pixel_format_float
  GLXEW_ATI_pixel_format_float = _glewSearchExtension("GLX_ATI_pixel_format_float", extStart, extEnd);
#endif /* GLX_ATI_pixel_format_float */
#ifdef GLX_ATI_render_texture
  GLXEW_ATI_render_texture = _glewSearchExtension("GLX_ATI_render_texture", extStart, extEnd);
  if (glewExperimental || GLXEW_ATI_render_texture) GLXEW_ATI_render_texture = !_glewInit_GLX_ATI_render_texture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_ATI_render_texture */
#ifdef GLX_EXT_buffer_age
  GLXEW_EXT_buffer_age = _glewSearchExtension("GLX_EXT_buffer_age", extStart, extEnd);
#endif /* GLX_EXT_buffer_age */
#ifdef GLX_EXT_create_context_es2_profile
  GLXEW_EXT_create_context_es2_profile = _glewSearchExtension("GLX_EXT_create_context_es2_profile", extStart, extEnd);
#endif /* GLX_EXT_create_context_es2_profile */
#ifdef GLX_EXT_create_context_es_profile
  GLXEW_EXT_create_context_es_profile = _glewSearchExtension("GLX_EXT_create_context_es_profile", extStart, extEnd);
#endif /* GLX_EXT_create_context_es_profile */
#ifdef GLX_EXT_fbconfig_packed_float
  GLXEW_EXT_fbconfig_packed_float = _glewSearchExtension("GLX_EXT_fbconfig_packed_float", extStart, extEnd);
#endif /* GLX_EXT_fbconfig_packed_float */
#ifdef GLX_EXT_framebuffer_sRGB
  GLXEW_EXT_framebuffer_sRGB = _glewSearchExtension("GLX_EXT_framebuffer_sRGB", extStart, extEnd);
#endif /* GLX_EXT_framebuffer_sRGB */
#ifdef GLX_EXT_import_context
  GLXEW_EXT_import_context = _glewSearchExtension("GLX_EXT_import_context", extStart, extEnd);
  if (glewExperimental || GLXEW_EXT_import_context) GLXEW_EXT_import_context = !_glewInit_GLX_EXT_import_context(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_EXT_import_context */
#ifdef GLX_EXT_scene_marker
  GLXEW_EXT_scene_marker = _glewSearchExtension("GLX_EXT_scene_marker", extStart, extEnd);
#endif /* GLX_EXT_scene_marker */
#ifdef GLX_EXT_stereo_tree
  GLXEW_EXT_stereo_tree = _glewSearchExtension("GLX_EXT_stereo_tree", extStart, extEnd);
#endif /* GLX_EXT_stereo_tree */
#ifdef GLX_EXT_swap_control
  GLXEW_EXT_swap_control = _glewSearchExtension("GLX_EXT_swap_control", extStart, extEnd);
  if (glewExperimental || GLXEW_EXT_swap_control) GLXEW_EXT_swap_control = !_glewInit_GLX_EXT_swap_control(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_EXT_swap_control */
#ifdef GLX_EXT_swap_control_tear
  GLXEW_EXT_swap_control_tear = _glewSearchExtension("GLX_EXT_swap_control_tear", extStart, extEnd);
#endif /* GLX_EXT_swap_control_tear */
#ifdef GLX_EXT_texture_from_pixmap
  GLXEW_EXT_texture_from_pixmap = _glewSearchExtension("GLX_EXT_texture_from_pixmap", extStart, extEnd);
  if (glewExperimental || GLXEW_EXT_texture_from_pixmap) GLXEW_EXT_texture_from_pixmap = !_glewInit_GLX_EXT_texture_from_pixmap(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_EXT_texture_from_pixmap */
#ifdef GLX_EXT_visual_info
  GLXEW_EXT_visual_info = _glewSearchExtension("GLX_EXT_visual_info", extStart, extEnd);
#endif /* GLX_EXT_visual_info */
#ifdef GLX_EXT_visual_rating
  GLXEW_EXT_visual_rating = _glewSearchExtension("GLX_EXT_visual_rating", extStart, extEnd);
#endif /* GLX_EXT_visual_rating */
#ifdef GLX_INTEL_swap_event
  GLXEW_INTEL_swap_event = _glewSearchExtension("GLX_INTEL_swap_event", extStart, extEnd);
#endif /* GLX_INTEL_swap_event */
#ifdef GLX_MESA_agp_offset
  GLXEW_MESA_agp_offset = _glewSearchExtension("GLX_MESA_agp_offset", extStart, extEnd);
  if (glewExperimental || GLXEW_MESA_agp_offset) GLXEW_MESA_agp_offset = !_glewInit_GLX_MESA_agp_offset(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_MESA_agp_offset */
#ifdef GLX_MESA_copy_sub_buffer
  GLXEW_MESA_copy_sub_buffer = _glewSearchExtension("GLX_MESA_copy_sub_buffer", extStart, extEnd);
  if (glewExperimental || GLXEW_MESA_copy_sub_buffer) GLXEW_MESA_copy_sub_buffer = !_glewInit_GLX_MESA_copy_sub_buffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_MESA_copy_sub_buffer */
#ifdef GLX_MESA_pixmap_colormap
  GLXEW_MESA_pixmap_colormap = _glewSearchExtension("GLX_MESA_pixmap_colormap", extStart, extEnd);
  if (glewExperimental || GLXEW_MESA_pixmap_colormap) GLXEW_MESA_pixmap_colormap = !_glewInit_GLX_MESA_pixmap_colormap(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_MESA_pixmap_colormap */
#ifdef GLX_MESA_query_renderer
  GLXEW_MESA_query_renderer = _glewSearchExtension("GLX_MESA_query_renderer", extStart, extEnd);
  if (glewExperimental || GLXEW_MESA_query_renderer) GLXEW_MESA_query_renderer = !_glewInit_GLX_MESA_query_renderer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_MESA_query_renderer */
#ifdef GLX_MESA_release_buffers
  GLXEW_MESA_release_buffers = _glewSearchExtension("GLX_MESA_release_buffers", extStart, extEnd);
  if (glewExperimental || GLXEW_MESA_release_buffers) GLXEW_MESA_release_buffers = !_glewInit_GLX_MESA_release_buffers(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_MESA_release_buffers */
#ifdef GLX_MESA_set_3dfx_mode
  GLXEW_MESA_set_3dfx_mode = _glewSearchExtension("GLX_MESA_set_3dfx_mode", extStart, extEnd);
  if (glewExperimental || GLXEW_MESA_set_3dfx_mode) GLXEW_MESA_set_3dfx_mode = !_glewInit_GLX_MESA_set_3dfx_mode(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_MESA_set_3dfx_mode */
#ifdef GLX_MESA_swap_control
  GLXEW_MESA_swap_control = _glewSearchExtension("GLX_MESA_swap_control", extStart, extEnd);
  if (glewExperimental || GLXEW_MESA_swap_control) GLXEW_MESA_swap_control = !_glewInit_GLX_MESA_swap_control(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_MESA_swap_control */
#ifdef GLX_NV_copy_buffer
  GLXEW_NV_copy_buffer = _glewSearchExtension("GLX_NV_copy_buffer", extStart, extEnd);
  if (glewExperimental || GLXEW_NV_copy_buffer) GLXEW_NV_copy_buffer = !_glewInit_GLX_NV_copy_buffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_NV_copy_buffer */
#ifdef GLX_NV_copy_image
  GLXEW_NV_copy_image = _glewSearchExtension("GLX_NV_copy_image", extStart, extEnd);
  if (glewExperimental || GLXEW_NV_copy_image) GLXEW_NV_copy_image = !_glewInit_GLX_NV_copy_image(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_NV_copy_image */
#ifdef GLX_NV_delay_before_swap
  GLXEW_NV_delay_before_swap = _glewSearchExtension("GLX_NV_delay_before_swap", extStart, extEnd);
  if (glewExperimental || GLXEW_NV_delay_before_swap) GLXEW_NV_delay_before_swap = !_glewInit_GLX_NV_delay_before_swap(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_NV_delay_before_swap */
#ifdef GLX_NV_float_buffer
  GLXEW_NV_float_buffer = _glewSearchExtension("GLX_NV_float_buffer", extStart, extEnd);
#endif /* GLX_NV_float_buffer */
#ifdef GLX_NV_multisample_coverage
  GLXEW_NV_multisample_coverage = _glewSearchExtension("GLX_NV_multisample_coverage", extStart, extEnd);
#endif /* GLX_NV_multisample_coverage */
#ifdef GLX_NV_present_video
  GLXEW_NV_present_video = _glewSearchExtension("GLX_NV_present_video", extStart, extEnd);
  if (glewExperimental || GLXEW_NV_present_video) GLXEW_NV_present_video = !_glewInit_GLX_NV_present_video(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_NV_present_video */
#ifdef GLX_NV_swap_group
  GLXEW_NV_swap_group = _glewSearchExtension("GLX_NV_swap_group", extStart, extEnd);
  if (glewExperimental || GLXEW_NV_swap_group) GLXEW_NV_swap_group = !_glewInit_GLX_NV_swap_group(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_NV_swap_group */
#ifdef GLX_NV_vertex_array_range
  GLXEW_NV_vertex_array_range = _glewSearchExtension("GLX_NV_vertex_array_range", extStart, extEnd);
  if (glewExperimental || GLXEW_NV_vertex_array_range) GLXEW_NV_vertex_array_range = !_glewInit_GLX_NV_vertex_array_range(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_NV_vertex_array_range */
#ifdef GLX_NV_video_capture
  GLXEW_NV_video_capture = _glewSearchExtension("GLX_NV_video_capture", extStart, extEnd);
  if (glewExperimental || GLXEW_NV_video_capture) GLXEW_NV_video_capture = !_glewInit_GLX_NV_video_capture(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_NV_video_capture */
#ifdef GLX_NV_video_out
  GLXEW_NV_video_out = _glewSearchExtension("GLX_NV_video_out", extStart, extEnd);
  if (glewExperimental || GLXEW_NV_video_out) GLXEW_NV_video_out = !_glewInit_GLX_NV_video_out(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_NV_video_out */
#ifdef GLX_OML_swap_method
  GLXEW_OML_swap_method = _glewSearchExtension("GLX_OML_swap_method", extStart, extEnd);
#endif /* GLX_OML_swap_method */
#ifdef GLX_OML_sync_control
  GLXEW_OML_sync_control = _glewSearchExtension("GLX_OML_sync_control", extStart, extEnd);
  if (glewExperimental || GLXEW_OML_sync_control) GLXEW_OML_sync_control = !_glewInit_GLX_OML_sync_control(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_OML_sync_control */
#ifdef GLX_SGIS_blended_overlay
  GLXEW_SGIS_blended_overlay = _glewSearchExtension("GLX_SGIS_blended_overlay", extStart, extEnd);
#endif /* GLX_SGIS_blended_overlay */
#ifdef GLX_SGIS_color_range
  GLXEW_SGIS_color_range = _glewSearchExtension("GLX_SGIS_color_range", extStart, extEnd);
#endif /* GLX_SGIS_color_range */
#ifdef GLX_SGIS_multisample
  GLXEW_SGIS_multisample = _glewSearchExtension("GLX_SGIS_multisample", extStart, extEnd);
#endif /* GLX_SGIS_multisample */
#ifdef GLX_SGIS_shared_multisample
  GLXEW_SGIS_shared_multisample = _glewSearchExtension("GLX_SGIS_shared_multisample", extStart, extEnd);
#endif /* GLX_SGIS_shared_multisample */
#ifdef GLX_SGIX_fbconfig
  GLXEW_SGIX_fbconfig = _glewSearchExtension("GLX_SGIX_fbconfig", extStart, extEnd);
  if (glewExperimental || GLXEW_SGIX_fbconfig) GLXEW_SGIX_fbconfig = !_glewInit_GLX_SGIX_fbconfig(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SGIX_fbconfig */
#ifdef GLX_SGIX_hyperpipe
  GLXEW_SGIX_hyperpipe = _glewSearchExtension("GLX_SGIX_hyperpipe", extStart, extEnd);
  if (glewExperimental || GLXEW_SGIX_hyperpipe) GLXEW_SGIX_hyperpipe = !_glewInit_GLX_SGIX_hyperpipe(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SGIX_hyperpipe */
#ifdef GLX_SGIX_pbuffer
  GLXEW_SGIX_pbuffer = _glewSearchExtension("GLX_SGIX_pbuffer", extStart, extEnd);
  if (glewExperimental || GLXEW_SGIX_pbuffer) GLXEW_SGIX_pbuffer = !_glewInit_GLX_SGIX_pbuffer(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SGIX_pbuffer */
#ifdef GLX_SGIX_swap_barrier
  GLXEW_SGIX_swap_barrier = _glewSearchExtension("GLX_SGIX_swap_barrier", extStart, extEnd);
  if (glewExperimental || GLXEW_SGIX_swap_barrier) GLXEW_SGIX_swap_barrier = !_glewInit_GLX_SGIX_swap_barrier(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SGIX_swap_barrier */
#ifdef GLX_SGIX_swap_group
  GLXEW_SGIX_swap_group = _glewSearchExtension("GLX_SGIX_swap_group", extStart, extEnd);
  if (glewExperimental || GLXEW_SGIX_swap_group) GLXEW_SGIX_swap_group = !_glewInit_GLX_SGIX_swap_group(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SGIX_swap_group */
#ifdef GLX_SGIX_video_resize
  GLXEW_SGIX_video_resize = _glewSearchExtension("GLX_SGIX_video_resize", extStart, extEnd);
  if (glewExperimental || GLXEW_SGIX_video_resize) GLXEW_SGIX_video_resize = !_glewInit_GLX_SGIX_video_resize(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SGIX_video_resize */
#ifdef GLX_SGIX_visual_select_group
  GLXEW_SGIX_visual_select_group = _glewSearchExtension("GLX_SGIX_visual_select_group", extStart, extEnd);
#endif /* GLX_SGIX_visual_select_group */
#ifdef GLX_SGI_cushion
  GLXEW_SGI_cushion = _glewSearchExtension("GLX_SGI_cushion", extStart, extEnd);
  if (glewExperimental || GLXEW_SGI_cushion) GLXEW_SGI_cushion = !_glewInit_GLX_SGI_cushion(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SGI_cushion */
#ifdef GLX_SGI_make_current_read
  GLXEW_SGI_make_current_read = _glewSearchExtension("GLX_SGI_make_current_read", extStart, extEnd);
  if (glewExperimental || GLXEW_SGI_make_current_read) GLXEW_SGI_make_current_read = !_glewInit_GLX_SGI_make_current_read(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SGI_make_current_read */
#ifdef GLX_SGI_swap_control
  GLXEW_SGI_swap_control = _glewSearchExtension("GLX_SGI_swap_control", extStart, extEnd);
  if (glewExperimental || GLXEW_SGI_swap_control) GLXEW_SGI_swap_control = !_glewInit_GLX_SGI_swap_control(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SGI_swap_control */
#ifdef GLX_SGI_video_sync
  GLXEW_SGI_video_sync = _glewSearchExtension("GLX_SGI_video_sync", extStart, extEnd);
  if (glewExperimental || GLXEW_SGI_video_sync) GLXEW_SGI_video_sync = !_glewInit_GLX_SGI_video_sync(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SGI_video_sync */
#ifdef GLX_SUN_get_transparent_index
  GLXEW_SUN_get_transparent_index = _glewSearchExtension("GLX_SUN_get_transparent_index", extStart, extEnd);
  if (glewExperimental || GLXEW_SUN_get_transparent_index) GLXEW_SUN_get_transparent_index = !_glewInit_GLX_SUN_get_transparent_index(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SUN_get_transparent_index */
#ifdef GLX_SUN_video_resize
  GLXEW_SUN_video_resize = _glewSearchExtension("GLX_SUN_video_resize", extStart, extEnd);
  if (glewExperimental || GLXEW_SUN_video_resize) GLXEW_SUN_video_resize = !_glewInit_GLX_SUN_video_resize(GLEW_CONTEXT_ARG_VAR_INIT);
#endif /* GLX_SUN_video_resize */

  return GLEW_OK;
}

#endif /* !defined(__ANDROID__) && !defined(__native_client__) && !defined(__HAIKU__) && (!defined(__APPLE__) || defined(GLEW_APPLE_GLX)) */

/* ------------------------------------------------------------------------ */

const GLubyte * GLEWAPIENTRY glewGetErrorString (GLenum error)
{
  static const GLubyte* _glewErrorString[] =
  {
    (const GLubyte*)"No error",
    (const GLubyte*)"Missing GL version",
    (const GLubyte*)"GL 1.1 and up are not supported",
    (const GLubyte*)"GLX 1.2 and up are not supported",
    (const GLubyte*)"Unknown error"
  };
  const size_t max_error = sizeof(_glewErrorString)/sizeof(*_glewErrorString) - 1;
  return _glewErrorString[(size_t)error > max_error ? max_error : (size_t)error];
}

const GLubyte * GLEWAPIENTRY glewGetString (GLenum name)
{
  static const GLubyte* _glewString[] =
  {
    (const GLubyte*)NULL,
    (const GLubyte*)"1.13.0",
    (const GLubyte*)"1",
    (const GLubyte*)"13",
    (const GLubyte*)"0"
  };
  const size_t max_string = sizeof(_glewString)/sizeof(*_glewString) - 1;
  return _glewString[(size_t)name > max_string ? 0 : (size_t)name];
}

/* ------------------------------------------------------------------------ */

GLboolean glewExperimental = GL_FALSE;

#if !defined(GLEW_MX)

GLenum GLEWAPIENTRY glewInit (void)
{
  GLenum r;
  r = glewContextInit();
  if ( r != 0 ) return r;
#if defined(_WIN32)
  return wglewInit();
#elif !defined(__ANDROID__) && !defined(__native_client__) && !defined(__HAIKU__) && (!defined(__APPLE__) || defined(GLEW_APPLE_GLX)) /* _UNIX */
  return glxewInit();
#else
  return r;
#endif /* _WIN32 */
}

#endif /* !GLEW_MX */
#ifdef GLEW_MX
GLboolean GLEWAPIENTRY glewContextIsSupported (const GLEWContext* ctx, const char* name)
#else
GLboolean GLEWAPIENTRY glewIsSupported (const char* name)
#endif
{
  const GLubyte* pos = (const GLubyte*)name;
  GLuint len = _glewStrLen(pos);
  GLboolean ret = GL_TRUE;
  while (ret && len > 0)
  {
    if (_glewStrSame1(&pos, &len, (const GLubyte*)"GL_", 3))
    {
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"VERSION_", 8))
      {
#ifdef GL_VERSION_1_2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"1_2", 3))
        {
          ret = GLEW_VERSION_1_2;
          continue;
        }
#endif
#ifdef GL_VERSION_1_2_1
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"1_2_1", 5))
        {
          ret = GLEW_VERSION_1_2_1;
          continue;
        }
#endif
#ifdef GL_VERSION_1_3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"1_3", 3))
        {
          ret = GLEW_VERSION_1_3;
          continue;
        }
#endif
#ifdef GL_VERSION_1_4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"1_4", 3))
        {
          ret = GLEW_VERSION_1_4;
          continue;
        }
#endif
#ifdef GL_VERSION_1_5
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"1_5", 3))
        {
          ret = GLEW_VERSION_1_5;
          continue;
        }
#endif
#ifdef GL_VERSION_2_0
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"2_0", 3))
        {
          ret = GLEW_VERSION_2_0;
          continue;
        }
#endif
#ifdef GL_VERSION_2_1
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"2_1", 3))
        {
          ret = GLEW_VERSION_2_1;
          continue;
        }
#endif
#ifdef GL_VERSION_3_0
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"3_0", 3))
        {
          ret = GLEW_VERSION_3_0;
          continue;
        }
#endif
#ifdef GL_VERSION_3_1
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"3_1", 3))
        {
          ret = GLEW_VERSION_3_1;
          continue;
        }
#endif
#ifdef GL_VERSION_3_2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"3_2", 3))
        {
          ret = GLEW_VERSION_3_2;
          continue;
        }
#endif
#ifdef GL_VERSION_3_3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"3_3", 3))
        {
          ret = GLEW_VERSION_3_3;
          continue;
        }
#endif
#ifdef GL_VERSION_4_0
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"4_0", 3))
        {
          ret = GLEW_VERSION_4_0;
          continue;
        }
#endif
#ifdef GL_VERSION_4_1
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"4_1", 3))
        {
          ret = GLEW_VERSION_4_1;
          continue;
        }
#endif
#ifdef GL_VERSION_4_2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"4_2", 3))
        {
          ret = GLEW_VERSION_4_2;
          continue;
        }
#endif
#ifdef GL_VERSION_4_3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"4_3", 3))
        {
          ret = GLEW_VERSION_4_3;
          continue;
        }
#endif
#ifdef GL_VERSION_4_4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"4_4", 3))
        {
          ret = GLEW_VERSION_4_4;
          continue;
        }
#endif
#ifdef GL_VERSION_4_5
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"4_5", 3))
        {
          ret = GLEW_VERSION_4_5;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"3DFX_", 5))
      {
#ifdef GL_3DFX_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample", 11))
        {
          ret = GLEW_3DFX_multisample;
          continue;
        }
#endif
#ifdef GL_3DFX_tbuffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"tbuffer", 7))
        {
          ret = GLEW_3DFX_tbuffer;
          continue;
        }
#endif
#ifdef GL_3DFX_texture_compression_FXT1
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_FXT1", 24))
        {
          ret = GLEW_3DFX_texture_compression_FXT1;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"AMD_", 4))
      {
#ifdef GL_AMD_blend_minmax_factor
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_minmax_factor", 19))
        {
          ret = GLEW_AMD_blend_minmax_factor;
          continue;
        }
#endif
#ifdef GL_AMD_conservative_depth
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"conservative_depth", 18))
        {
          ret = GLEW_AMD_conservative_depth;
          continue;
        }
#endif
#ifdef GL_AMD_debug_output
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"debug_output", 12))
        {
          ret = GLEW_AMD_debug_output;
          continue;
        }
#endif
#ifdef GL_AMD_depth_clamp_separate
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_clamp_separate", 20))
        {
          ret = GLEW_AMD_depth_clamp_separate;
          continue;
        }
#endif
#ifdef GL_AMD_draw_buffers_blend
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_buffers_blend", 18))
        {
          ret = GLEW_AMD_draw_buffers_blend;
          continue;
        }
#endif
#ifdef GL_AMD_gcn_shader
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gcn_shader", 10))
        {
          ret = GLEW_AMD_gcn_shader;
          continue;
        }
#endif
#ifdef GL_AMD_gpu_shader_int64
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_shader_int64", 16))
        {
          ret = GLEW_AMD_gpu_shader_int64;
          continue;
        }
#endif
#ifdef GL_AMD_interleaved_elements
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"interleaved_elements", 20))
        {
          ret = GLEW_AMD_interleaved_elements;
          continue;
        }
#endif
#ifdef GL_AMD_multi_draw_indirect
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multi_draw_indirect", 19))
        {
          ret = GLEW_AMD_multi_draw_indirect;
          continue;
        }
#endif
#ifdef GL_AMD_name_gen_delete
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"name_gen_delete", 15))
        {
          ret = GLEW_AMD_name_gen_delete;
          continue;
        }
#endif
#ifdef GL_AMD_occlusion_query_event
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"occlusion_query_event", 21))
        {
          ret = GLEW_AMD_occlusion_query_event;
          continue;
        }
#endif
#ifdef GL_AMD_performance_monitor
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"performance_monitor", 19))
        {
          ret = GLEW_AMD_performance_monitor;
          continue;
        }
#endif
#ifdef GL_AMD_pinned_memory
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pinned_memory", 13))
        {
          ret = GLEW_AMD_pinned_memory;
          continue;
        }
#endif
#ifdef GL_AMD_query_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"query_buffer_object", 19))
        {
          ret = GLEW_AMD_query_buffer_object;
          continue;
        }
#endif
#ifdef GL_AMD_sample_positions
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sample_positions", 16))
        {
          ret = GLEW_AMD_sample_positions;
          continue;
        }
#endif
#ifdef GL_AMD_seamless_cubemap_per_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"seamless_cubemap_per_texture", 28))
        {
          ret = GLEW_AMD_seamless_cubemap_per_texture;
          continue;
        }
#endif
#ifdef GL_AMD_shader_atomic_counter_ops
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_atomic_counter_ops", 25))
        {
          ret = GLEW_AMD_shader_atomic_counter_ops;
          continue;
        }
#endif
#ifdef GL_AMD_shader_stencil_export
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_stencil_export", 21))
        {
          ret = GLEW_AMD_shader_stencil_export;
          continue;
        }
#endif
#ifdef GL_AMD_shader_stencil_value_export
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_stencil_value_export", 27))
        {
          ret = GLEW_AMD_shader_stencil_value_export;
          continue;
        }
#endif
#ifdef GL_AMD_shader_trinary_minmax
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_trinary_minmax", 21))
        {
          ret = GLEW_AMD_shader_trinary_minmax;
          continue;
        }
#endif
#ifdef GL_AMD_sparse_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sparse_texture", 14))
        {
          ret = GLEW_AMD_sparse_texture;
          continue;
        }
#endif
#ifdef GL_AMD_stencil_operation_extended
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"stencil_operation_extended", 26))
        {
          ret = GLEW_AMD_stencil_operation_extended;
          continue;
        }
#endif
#ifdef GL_AMD_texture_texture4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_texture4", 16))
        {
          ret = GLEW_AMD_texture_texture4;
          continue;
        }
#endif
#ifdef GL_AMD_transform_feedback3_lines_triangles
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transform_feedback3_lines_triangles", 35))
        {
          ret = GLEW_AMD_transform_feedback3_lines_triangles;
          continue;
        }
#endif
#ifdef GL_AMD_transform_feedback4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transform_feedback4", 19))
        {
          ret = GLEW_AMD_transform_feedback4;
          continue;
        }
#endif
#ifdef GL_AMD_vertex_shader_layer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_shader_layer", 19))
        {
          ret = GLEW_AMD_vertex_shader_layer;
          continue;
        }
#endif
#ifdef GL_AMD_vertex_shader_tessellator
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_shader_tessellator", 25))
        {
          ret = GLEW_AMD_vertex_shader_tessellator;
          continue;
        }
#endif
#ifdef GL_AMD_vertex_shader_viewport_index
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_shader_viewport_index", 28))
        {
          ret = GLEW_AMD_vertex_shader_viewport_index;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"ANGLE_", 6))
      {
#ifdef GL_ANGLE_depth_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_texture", 13))
        {
          ret = GLEW_ANGLE_depth_texture;
          continue;
        }
#endif
#ifdef GL_ANGLE_framebuffer_blit
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_blit", 16))
        {
          ret = GLEW_ANGLE_framebuffer_blit;
          continue;
        }
#endif
#ifdef GL_ANGLE_framebuffer_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_multisample", 23))
        {
          ret = GLEW_ANGLE_framebuffer_multisample;
          continue;
        }
#endif
#ifdef GL_ANGLE_instanced_arrays
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"instanced_arrays", 16))
        {
          ret = GLEW_ANGLE_instanced_arrays;
          continue;
        }
#endif
#ifdef GL_ANGLE_pack_reverse_row_order
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pack_reverse_row_order", 22))
        {
          ret = GLEW_ANGLE_pack_reverse_row_order;
          continue;
        }
#endif
#ifdef GL_ANGLE_program_binary
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"program_binary", 14))
        {
          ret = GLEW_ANGLE_program_binary;
          continue;
        }
#endif
#ifdef GL_ANGLE_texture_compression_dxt1
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_dxt1", 24))
        {
          ret = GLEW_ANGLE_texture_compression_dxt1;
          continue;
        }
#endif
#ifdef GL_ANGLE_texture_compression_dxt3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_dxt3", 24))
        {
          ret = GLEW_ANGLE_texture_compression_dxt3;
          continue;
        }
#endif
#ifdef GL_ANGLE_texture_compression_dxt5
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_dxt5", 24))
        {
          ret = GLEW_ANGLE_texture_compression_dxt5;
          continue;
        }
#endif
#ifdef GL_ANGLE_texture_usage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_usage", 13))
        {
          ret = GLEW_ANGLE_texture_usage;
          continue;
        }
#endif
#ifdef GL_ANGLE_timer_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"timer_query", 11))
        {
          ret = GLEW_ANGLE_timer_query;
          continue;
        }
#endif
#ifdef GL_ANGLE_translated_shader_source
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"translated_shader_source", 24))
        {
          ret = GLEW_ANGLE_translated_shader_source;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"APPLE_", 6))
      {
#ifdef GL_APPLE_aux_depth_stencil
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"aux_depth_stencil", 17))
        {
          ret = GLEW_APPLE_aux_depth_stencil;
          continue;
        }
#endif
#ifdef GL_APPLE_client_storage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"client_storage", 14))
        {
          ret = GLEW_APPLE_client_storage;
          continue;
        }
#endif
#ifdef GL_APPLE_element_array
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"element_array", 13))
        {
          ret = GLEW_APPLE_element_array;
          continue;
        }
#endif
#ifdef GL_APPLE_fence
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fence", 5))
        {
          ret = GLEW_APPLE_fence;
          continue;
        }
#endif
#ifdef GL_APPLE_float_pixels
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"float_pixels", 12))
        {
          ret = GLEW_APPLE_float_pixels;
          continue;
        }
#endif
#ifdef GL_APPLE_flush_buffer_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"flush_buffer_range", 18))
        {
          ret = GLEW_APPLE_flush_buffer_range;
          continue;
        }
#endif
#ifdef GL_APPLE_object_purgeable
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"object_purgeable", 16))
        {
          ret = GLEW_APPLE_object_purgeable;
          continue;
        }
#endif
#ifdef GL_APPLE_pixel_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_buffer", 12))
        {
          ret = GLEW_APPLE_pixel_buffer;
          continue;
        }
#endif
#ifdef GL_APPLE_rgb_422
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"rgb_422", 7))
        {
          ret = GLEW_APPLE_rgb_422;
          continue;
        }
#endif
#ifdef GL_APPLE_row_bytes
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"row_bytes", 9))
        {
          ret = GLEW_APPLE_row_bytes;
          continue;
        }
#endif
#ifdef GL_APPLE_specular_vector
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"specular_vector", 15))
        {
          ret = GLEW_APPLE_specular_vector;
          continue;
        }
#endif
#ifdef GL_APPLE_texture_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_range", 13))
        {
          ret = GLEW_APPLE_texture_range;
          continue;
        }
#endif
#ifdef GL_APPLE_transform_hint
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transform_hint", 14))
        {
          ret = GLEW_APPLE_transform_hint;
          continue;
        }
#endif
#ifdef GL_APPLE_vertex_array_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_object", 19))
        {
          ret = GLEW_APPLE_vertex_array_object;
          continue;
        }
#endif
#ifdef GL_APPLE_vertex_array_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_range", 18))
        {
          ret = GLEW_APPLE_vertex_array_range;
          continue;
        }
#endif
#ifdef GL_APPLE_vertex_program_evaluators
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_program_evaluators", 25))
        {
          ret = GLEW_APPLE_vertex_program_evaluators;
          continue;
        }
#endif
#ifdef GL_APPLE_ycbcr_422
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"ycbcr_422", 9))
        {
          ret = GLEW_APPLE_ycbcr_422;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"ARB_", 4))
      {
#ifdef GL_ARB_ES2_compatibility
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"ES2_compatibility", 17))
        {
          ret = GLEW_ARB_ES2_compatibility;
          continue;
        }
#endif
#ifdef GL_ARB_ES3_1_compatibility
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"ES3_1_compatibility", 19))
        {
          ret = GLEW_ARB_ES3_1_compatibility;
          continue;
        }
#endif
#ifdef GL_ARB_ES3_2_compatibility
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"ES3_2_compatibility", 19))
        {
          ret = GLEW_ARB_ES3_2_compatibility;
          continue;
        }
#endif
#ifdef GL_ARB_ES3_compatibility
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"ES3_compatibility", 17))
        {
          ret = GLEW_ARB_ES3_compatibility;
          continue;
        }
#endif
#ifdef GL_ARB_arrays_of_arrays
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"arrays_of_arrays", 16))
        {
          ret = GLEW_ARB_arrays_of_arrays;
          continue;
        }
#endif
#ifdef GL_ARB_base_instance
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"base_instance", 13))
        {
          ret = GLEW_ARB_base_instance;
          continue;
        }
#endif
#ifdef GL_ARB_bindless_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"bindless_texture", 16))
        {
          ret = GLEW_ARB_bindless_texture;
          continue;
        }
#endif
#ifdef GL_ARB_blend_func_extended
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_func_extended", 19))
        {
          ret = GLEW_ARB_blend_func_extended;
          continue;
        }
#endif
#ifdef GL_ARB_buffer_storage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"buffer_storage", 14))
        {
          ret = GLEW_ARB_buffer_storage;
          continue;
        }
#endif
#ifdef GL_ARB_cl_event
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"cl_event", 8))
        {
          ret = GLEW_ARB_cl_event;
          continue;
        }
#endif
#ifdef GL_ARB_clear_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"clear_buffer_object", 19))
        {
          ret = GLEW_ARB_clear_buffer_object;
          continue;
        }
#endif
#ifdef GL_ARB_clear_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"clear_texture", 13))
        {
          ret = GLEW_ARB_clear_texture;
          continue;
        }
#endif
#ifdef GL_ARB_clip_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"clip_control", 12))
        {
          ret = GLEW_ARB_clip_control;
          continue;
        }
#endif
#ifdef GL_ARB_color_buffer_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"color_buffer_float", 18))
        {
          ret = GLEW_ARB_color_buffer_float;
          continue;
        }
#endif
#ifdef GL_ARB_compatibility
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"compatibility", 13))
        {
          ret = GLEW_ARB_compatibility;
          continue;
        }
#endif
#ifdef GL_ARB_compressed_texture_pixel_storage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"compressed_texture_pixel_storage", 32))
        {
          ret = GLEW_ARB_compressed_texture_pixel_storage;
          continue;
        }
#endif
#ifdef GL_ARB_compute_shader
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"compute_shader", 14))
        {
          ret = GLEW_ARB_compute_shader;
          continue;
        }
#endif
#ifdef GL_ARB_compute_variable_group_size
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"compute_variable_group_size", 27))
        {
          ret = GLEW_ARB_compute_variable_group_size;
          continue;
        }
#endif
#ifdef GL_ARB_conditional_render_inverted
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"conditional_render_inverted", 27))
        {
          ret = GLEW_ARB_conditional_render_inverted;
          continue;
        }
#endif
#ifdef GL_ARB_conservative_depth
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"conservative_depth", 18))
        {
          ret = GLEW_ARB_conservative_depth;
          continue;
        }
#endif
#ifdef GL_ARB_copy_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"copy_buffer", 11))
        {
          ret = GLEW_ARB_copy_buffer;
          continue;
        }
#endif
#ifdef GL_ARB_copy_image
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"copy_image", 10))
        {
          ret = GLEW_ARB_copy_image;
          continue;
        }
#endif
#ifdef GL_ARB_cull_distance
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"cull_distance", 13))
        {
          ret = GLEW_ARB_cull_distance;
          continue;
        }
#endif
#ifdef GL_ARB_debug_output
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"debug_output", 12))
        {
          ret = GLEW_ARB_debug_output;
          continue;
        }
#endif
#ifdef GL_ARB_depth_buffer_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_buffer_float", 18))
        {
          ret = GLEW_ARB_depth_buffer_float;
          continue;
        }
#endif
#ifdef GL_ARB_depth_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_clamp", 11))
        {
          ret = GLEW_ARB_depth_clamp;
          continue;
        }
#endif
#ifdef GL_ARB_depth_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_texture", 13))
        {
          ret = GLEW_ARB_depth_texture;
          continue;
        }
#endif
#ifdef GL_ARB_derivative_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"derivative_control", 18))
        {
          ret = GLEW_ARB_derivative_control;
          continue;
        }
#endif
#ifdef GL_ARB_direct_state_access
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"direct_state_access", 19))
        {
          ret = GLEW_ARB_direct_state_access;
          continue;
        }
#endif
#ifdef GL_ARB_draw_buffers
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_buffers", 12))
        {
          ret = GLEW_ARB_draw_buffers;
          continue;
        }
#endif
#ifdef GL_ARB_draw_buffers_blend
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_buffers_blend", 18))
        {
          ret = GLEW_ARB_draw_buffers_blend;
          continue;
        }
#endif
#ifdef GL_ARB_draw_elements_base_vertex
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_elements_base_vertex", 25))
        {
          ret = GLEW_ARB_draw_elements_base_vertex;
          continue;
        }
#endif
#ifdef GL_ARB_draw_indirect
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_indirect", 13))
        {
          ret = GLEW_ARB_draw_indirect;
          continue;
        }
#endif
#ifdef GL_ARB_draw_instanced
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_instanced", 14))
        {
          ret = GLEW_ARB_draw_instanced;
          continue;
        }
#endif
#ifdef GL_ARB_enhanced_layouts
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"enhanced_layouts", 16))
        {
          ret = GLEW_ARB_enhanced_layouts;
          continue;
        }
#endif
#ifdef GL_ARB_explicit_attrib_location
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"explicit_attrib_location", 24))
        {
          ret = GLEW_ARB_explicit_attrib_location;
          continue;
        }
#endif
#ifdef GL_ARB_explicit_uniform_location
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"explicit_uniform_location", 25))
        {
          ret = GLEW_ARB_explicit_uniform_location;
          continue;
        }
#endif
#ifdef GL_ARB_fragment_coord_conventions
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_coord_conventions", 26))
        {
          ret = GLEW_ARB_fragment_coord_conventions;
          continue;
        }
#endif
#ifdef GL_ARB_fragment_layer_viewport
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_layer_viewport", 23))
        {
          ret = GLEW_ARB_fragment_layer_viewport;
          continue;
        }
#endif
#ifdef GL_ARB_fragment_program
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_program", 16))
        {
          ret = GLEW_ARB_fragment_program;
          continue;
        }
#endif
#ifdef GL_ARB_fragment_program_shadow
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_program_shadow", 23))
        {
          ret = GLEW_ARB_fragment_program_shadow;
          continue;
        }
#endif
#ifdef GL_ARB_fragment_shader
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_shader", 15))
        {
          ret = GLEW_ARB_fragment_shader;
          continue;
        }
#endif
#ifdef GL_ARB_fragment_shader_interlock
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_shader_interlock", 25))
        {
          ret = GLEW_ARB_fragment_shader_interlock;
          continue;
        }
#endif
#ifdef GL_ARB_framebuffer_no_attachments
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_no_attachments", 26))
        {
          ret = GLEW_ARB_framebuffer_no_attachments;
          continue;
        }
#endif
#ifdef GL_ARB_framebuffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_object", 18))
        {
          ret = GLEW_ARB_framebuffer_object;
          continue;
        }
#endif
#ifdef GL_ARB_framebuffer_sRGB
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_sRGB", 16))
        {
          ret = GLEW_ARB_framebuffer_sRGB;
          continue;
        }
#endif
#ifdef GL_ARB_geometry_shader4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"geometry_shader4", 16))
        {
          ret = GLEW_ARB_geometry_shader4;
          continue;
        }
#endif
#ifdef GL_ARB_get_program_binary
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"get_program_binary", 18))
        {
          ret = GLEW_ARB_get_program_binary;
          continue;
        }
#endif
#ifdef GL_ARB_get_texture_sub_image
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"get_texture_sub_image", 21))
        {
          ret = GLEW_ARB_get_texture_sub_image;
          continue;
        }
#endif
#ifdef GL_ARB_gpu_shader5
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_shader5", 11))
        {
          ret = GLEW_ARB_gpu_shader5;
          continue;
        }
#endif
#ifdef GL_ARB_gpu_shader_fp64
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_shader_fp64", 15))
        {
          ret = GLEW_ARB_gpu_shader_fp64;
          continue;
        }
#endif
#ifdef GL_ARB_gpu_shader_int64
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_shader_int64", 16))
        {
          ret = GLEW_ARB_gpu_shader_int64;
          continue;
        }
#endif
#ifdef GL_ARB_half_float_pixel
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"half_float_pixel", 16))
        {
          ret = GLEW_ARB_half_float_pixel;
          continue;
        }
#endif
#ifdef GL_ARB_half_float_vertex
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"half_float_vertex", 17))
        {
          ret = GLEW_ARB_half_float_vertex;
          continue;
        }
#endif
#ifdef GL_ARB_imaging
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"imaging", 7))
        {
          ret = GLEW_ARB_imaging;
          continue;
        }
#endif
#ifdef GL_ARB_indirect_parameters
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"indirect_parameters", 19))
        {
          ret = GLEW_ARB_indirect_parameters;
          continue;
        }
#endif
#ifdef GL_ARB_instanced_arrays
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"instanced_arrays", 16))
        {
          ret = GLEW_ARB_instanced_arrays;
          continue;
        }
#endif
#ifdef GL_ARB_internalformat_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"internalformat_query", 20))
        {
          ret = GLEW_ARB_internalformat_query;
          continue;
        }
#endif
#ifdef GL_ARB_internalformat_query2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"internalformat_query2", 21))
        {
          ret = GLEW_ARB_internalformat_query2;
          continue;
        }
#endif
#ifdef GL_ARB_invalidate_subdata
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"invalidate_subdata", 18))
        {
          ret = GLEW_ARB_invalidate_subdata;
          continue;
        }
#endif
#ifdef GL_ARB_map_buffer_alignment
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"map_buffer_alignment", 20))
        {
          ret = GLEW_ARB_map_buffer_alignment;
          continue;
        }
#endif
#ifdef GL_ARB_map_buffer_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"map_buffer_range", 16))
        {
          ret = GLEW_ARB_map_buffer_range;
          continue;
        }
#endif
#ifdef GL_ARB_matrix_palette
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"matrix_palette", 14))
        {
          ret = GLEW_ARB_matrix_palette;
          continue;
        }
#endif
#ifdef GL_ARB_multi_bind
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multi_bind", 10))
        {
          ret = GLEW_ARB_multi_bind;
          continue;
        }
#endif
#ifdef GL_ARB_multi_draw_indirect
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multi_draw_indirect", 19))
        {
          ret = GLEW_ARB_multi_draw_indirect;
          continue;
        }
#endif
#ifdef GL_ARB_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample", 11))
        {
          ret = GLEW_ARB_multisample;
          continue;
        }
#endif
#ifdef GL_ARB_multitexture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multitexture", 12))
        {
          ret = GLEW_ARB_multitexture;
          continue;
        }
#endif
#ifdef GL_ARB_occlusion_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"occlusion_query", 15))
        {
          ret = GLEW_ARB_occlusion_query;
          continue;
        }
#endif
#ifdef GL_ARB_occlusion_query2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"occlusion_query2", 16))
        {
          ret = GLEW_ARB_occlusion_query2;
          continue;
        }
#endif
#ifdef GL_ARB_parallel_shader_compile
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"parallel_shader_compile", 23))
        {
          ret = GLEW_ARB_parallel_shader_compile;
          continue;
        }
#endif
#ifdef GL_ARB_pipeline_statistics_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pipeline_statistics_query", 25))
        {
          ret = GLEW_ARB_pipeline_statistics_query;
          continue;
        }
#endif
#ifdef GL_ARB_pixel_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_buffer_object", 19))
        {
          ret = GLEW_ARB_pixel_buffer_object;
          continue;
        }
#endif
#ifdef GL_ARB_point_parameters
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"point_parameters", 16))
        {
          ret = GLEW_ARB_point_parameters;
          continue;
        }
#endif
#ifdef GL_ARB_point_sprite
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"point_sprite", 12))
        {
          ret = GLEW_ARB_point_sprite;
          continue;
        }
#endif
#ifdef GL_ARB_post_depth_coverage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"post_depth_coverage", 19))
        {
          ret = GLEW_ARB_post_depth_coverage;
          continue;
        }
#endif
#ifdef GL_ARB_program_interface_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"program_interface_query", 23))
        {
          ret = GLEW_ARB_program_interface_query;
          continue;
        }
#endif
#ifdef GL_ARB_provoking_vertex
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"provoking_vertex", 16))
        {
          ret = GLEW_ARB_provoking_vertex;
          continue;
        }
#endif
#ifdef GL_ARB_query_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"query_buffer_object", 19))
        {
          ret = GLEW_ARB_query_buffer_object;
          continue;
        }
#endif
#ifdef GL_ARB_robust_buffer_access_behavior
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"robust_buffer_access_behavior", 29))
        {
          ret = GLEW_ARB_robust_buffer_access_behavior;
          continue;
        }
#endif
#ifdef GL_ARB_robustness
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"robustness", 10))
        {
          ret = GLEW_ARB_robustness;
          continue;
        }
#endif
#ifdef GL_ARB_robustness_application_isolation
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"robustness_application_isolation", 32))
        {
          ret = GLEW_ARB_robustness_application_isolation;
          continue;
        }
#endif
#ifdef GL_ARB_robustness_share_group_isolation
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"robustness_share_group_isolation", 32))
        {
          ret = GLEW_ARB_robustness_share_group_isolation;
          continue;
        }
#endif
#ifdef GL_ARB_sample_locations
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sample_locations", 16))
        {
          ret = GLEW_ARB_sample_locations;
          continue;
        }
#endif
#ifdef GL_ARB_sample_shading
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sample_shading", 14))
        {
          ret = GLEW_ARB_sample_shading;
          continue;
        }
#endif
#ifdef GL_ARB_sampler_objects
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sampler_objects", 15))
        {
          ret = GLEW_ARB_sampler_objects;
          continue;
        }
#endif
#ifdef GL_ARB_seamless_cube_map
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"seamless_cube_map", 17))
        {
          ret = GLEW_ARB_seamless_cube_map;
          continue;
        }
#endif
#ifdef GL_ARB_seamless_cubemap_per_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"seamless_cubemap_per_texture", 28))
        {
          ret = GLEW_ARB_seamless_cubemap_per_texture;
          continue;
        }
#endif
#ifdef GL_ARB_separate_shader_objects
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"separate_shader_objects", 23))
        {
          ret = GLEW_ARB_separate_shader_objects;
          continue;
        }
#endif
#ifdef GL_ARB_shader_atomic_counter_ops
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_atomic_counter_ops", 25))
        {
          ret = GLEW_ARB_shader_atomic_counter_ops;
          continue;
        }
#endif
#ifdef GL_ARB_shader_atomic_counters
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_atomic_counters", 22))
        {
          ret = GLEW_ARB_shader_atomic_counters;
          continue;
        }
#endif
#ifdef GL_ARB_shader_ballot
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_ballot", 13))
        {
          ret = GLEW_ARB_shader_ballot;
          continue;
        }
#endif
#ifdef GL_ARB_shader_bit_encoding
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_bit_encoding", 19))
        {
          ret = GLEW_ARB_shader_bit_encoding;
          continue;
        }
#endif
#ifdef GL_ARB_shader_clock
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_clock", 12))
        {
          ret = GLEW_ARB_shader_clock;
          continue;
        }
#endif
#ifdef GL_ARB_shader_draw_parameters
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_draw_parameters", 22))
        {
          ret = GLEW_ARB_shader_draw_parameters;
          continue;
        }
#endif
#ifdef GL_ARB_shader_group_vote
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_group_vote", 17))
        {
          ret = GLEW_ARB_shader_group_vote;
          continue;
        }
#endif
#ifdef GL_ARB_shader_image_load_store
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_image_load_store", 23))
        {
          ret = GLEW_ARB_shader_image_load_store;
          continue;
        }
#endif
#ifdef GL_ARB_shader_image_size
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_image_size", 17))
        {
          ret = GLEW_ARB_shader_image_size;
          continue;
        }
#endif
#ifdef GL_ARB_shader_objects
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_objects", 14))
        {
          ret = GLEW_ARB_shader_objects;
          continue;
        }
#endif
#ifdef GL_ARB_shader_precision
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_precision", 16))
        {
          ret = GLEW_ARB_shader_precision;
          continue;
        }
#endif
#ifdef GL_ARB_shader_stencil_export
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_stencil_export", 21))
        {
          ret = GLEW_ARB_shader_stencil_export;
          continue;
        }
#endif
#ifdef GL_ARB_shader_storage_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_storage_buffer_object", 28))
        {
          ret = GLEW_ARB_shader_storage_buffer_object;
          continue;
        }
#endif
#ifdef GL_ARB_shader_subroutine
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_subroutine", 17))
        {
          ret = GLEW_ARB_shader_subroutine;
          continue;
        }
#endif
#ifdef GL_ARB_shader_texture_image_samples
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_texture_image_samples", 28))
        {
          ret = GLEW_ARB_shader_texture_image_samples;
          continue;
        }
#endif
#ifdef GL_ARB_shader_texture_lod
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_texture_lod", 18))
        {
          ret = GLEW_ARB_shader_texture_lod;
          continue;
        }
#endif
#ifdef GL_ARB_shader_viewport_layer_array
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_viewport_layer_array", 27))
        {
          ret = GLEW_ARB_shader_viewport_layer_array;
          continue;
        }
#endif
#ifdef GL_ARB_shading_language_100
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shading_language_100", 20))
        {
          ret = GLEW_ARB_shading_language_100;
          continue;
        }
#endif
#ifdef GL_ARB_shading_language_420pack
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shading_language_420pack", 24))
        {
          ret = GLEW_ARB_shading_language_420pack;
          continue;
        }
#endif
#ifdef GL_ARB_shading_language_include
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shading_language_include", 24))
        {
          ret = GLEW_ARB_shading_language_include;
          continue;
        }
#endif
#ifdef GL_ARB_shading_language_packing
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shading_language_packing", 24))
        {
          ret = GLEW_ARB_shading_language_packing;
          continue;
        }
#endif
#ifdef GL_ARB_shadow
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shadow", 6))
        {
          ret = GLEW_ARB_shadow;
          continue;
        }
#endif
#ifdef GL_ARB_shadow_ambient
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shadow_ambient", 14))
        {
          ret = GLEW_ARB_shadow_ambient;
          continue;
        }
#endif
#ifdef GL_ARB_sparse_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sparse_buffer", 13))
        {
          ret = GLEW_ARB_sparse_buffer;
          continue;
        }
#endif
#ifdef GL_ARB_sparse_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sparse_texture", 14))
        {
          ret = GLEW_ARB_sparse_texture;
          continue;
        }
#endif
#ifdef GL_ARB_sparse_texture2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sparse_texture2", 15))
        {
          ret = GLEW_ARB_sparse_texture2;
          continue;
        }
#endif
#ifdef GL_ARB_sparse_texture_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sparse_texture_clamp", 20))
        {
          ret = GLEW_ARB_sparse_texture_clamp;
          continue;
        }
#endif
#ifdef GL_ARB_stencil_texturing
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"stencil_texturing", 17))
        {
          ret = GLEW_ARB_stencil_texturing;
          continue;
        }
#endif
#ifdef GL_ARB_sync
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sync", 4))
        {
          ret = GLEW_ARB_sync;
          continue;
        }
#endif
#ifdef GL_ARB_tessellation_shader
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"tessellation_shader", 19))
        {
          ret = GLEW_ARB_tessellation_shader;
          continue;
        }
#endif
#ifdef GL_ARB_texture_barrier
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_barrier", 15))
        {
          ret = GLEW_ARB_texture_barrier;
          continue;
        }
#endif
#ifdef GL_ARB_texture_border_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_border_clamp", 20))
        {
          ret = GLEW_ARB_texture_border_clamp;
          continue;
        }
#endif
#ifdef GL_ARB_texture_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_buffer_object", 21))
        {
          ret = GLEW_ARB_texture_buffer_object;
          continue;
        }
#endif
#ifdef GL_ARB_texture_buffer_object_rgb32
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_buffer_object_rgb32", 27))
        {
          ret = GLEW_ARB_texture_buffer_object_rgb32;
          continue;
        }
#endif
#ifdef GL_ARB_texture_buffer_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_buffer_range", 20))
        {
          ret = GLEW_ARB_texture_buffer_range;
          continue;
        }
#endif
#ifdef GL_ARB_texture_compression
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression", 19))
        {
          ret = GLEW_ARB_texture_compression;
          continue;
        }
#endif
#ifdef GL_ARB_texture_compression_bptc
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_bptc", 24))
        {
          ret = GLEW_ARB_texture_compression_bptc;
          continue;
        }
#endif
#ifdef GL_ARB_texture_compression_rgtc
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_rgtc", 24))
        {
          ret = GLEW_ARB_texture_compression_rgtc;
          continue;
        }
#endif
#ifdef GL_ARB_texture_cube_map
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_cube_map", 16))
        {
          ret = GLEW_ARB_texture_cube_map;
          continue;
        }
#endif
#ifdef GL_ARB_texture_cube_map_array
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_cube_map_array", 22))
        {
          ret = GLEW_ARB_texture_cube_map_array;
          continue;
        }
#endif
#ifdef GL_ARB_texture_env_add
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_add", 15))
        {
          ret = GLEW_ARB_texture_env_add;
          continue;
        }
#endif
#ifdef GL_ARB_texture_env_combine
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_combine", 19))
        {
          ret = GLEW_ARB_texture_env_combine;
          continue;
        }
#endif
#ifdef GL_ARB_texture_env_crossbar
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_crossbar", 20))
        {
          ret = GLEW_ARB_texture_env_crossbar;
          continue;
        }
#endif
#ifdef GL_ARB_texture_env_dot3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_dot3", 16))
        {
          ret = GLEW_ARB_texture_env_dot3;
          continue;
        }
#endif
#ifdef GL_ARB_texture_filter_minmax
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_filter_minmax", 21))
        {
          ret = GLEW_ARB_texture_filter_minmax;
          continue;
        }
#endif
#ifdef GL_ARB_texture_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_float", 13))
        {
          ret = GLEW_ARB_texture_float;
          continue;
        }
#endif
#ifdef GL_ARB_texture_gather
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_gather", 14))
        {
          ret = GLEW_ARB_texture_gather;
          continue;
        }
#endif
#ifdef GL_ARB_texture_mirror_clamp_to_edge
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_mirror_clamp_to_edge", 28))
        {
          ret = GLEW_ARB_texture_mirror_clamp_to_edge;
          continue;
        }
#endif
#ifdef GL_ARB_texture_mirrored_repeat
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_mirrored_repeat", 23))
        {
          ret = GLEW_ARB_texture_mirrored_repeat;
          continue;
        }
#endif
#ifdef GL_ARB_texture_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_multisample", 19))
        {
          ret = GLEW_ARB_texture_multisample;
          continue;
        }
#endif
#ifdef GL_ARB_texture_non_power_of_two
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_non_power_of_two", 24))
        {
          ret = GLEW_ARB_texture_non_power_of_two;
          continue;
        }
#endif
#ifdef GL_ARB_texture_query_levels
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_query_levels", 20))
        {
          ret = GLEW_ARB_texture_query_levels;
          continue;
        }
#endif
#ifdef GL_ARB_texture_query_lod
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_query_lod", 17))
        {
          ret = GLEW_ARB_texture_query_lod;
          continue;
        }
#endif
#ifdef GL_ARB_texture_rectangle
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_rectangle", 17))
        {
          ret = GLEW_ARB_texture_rectangle;
          continue;
        }
#endif
#ifdef GL_ARB_texture_rg
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_rg", 10))
        {
          ret = GLEW_ARB_texture_rg;
          continue;
        }
#endif
#ifdef GL_ARB_texture_rgb10_a2ui
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_rgb10_a2ui", 18))
        {
          ret = GLEW_ARB_texture_rgb10_a2ui;
          continue;
        }
#endif
#ifdef GL_ARB_texture_stencil8
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_stencil8", 16))
        {
          ret = GLEW_ARB_texture_stencil8;
          continue;
        }
#endif
#ifdef GL_ARB_texture_storage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_storage", 15))
        {
          ret = GLEW_ARB_texture_storage;
          continue;
        }
#endif
#ifdef GL_ARB_texture_storage_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_storage_multisample", 27))
        {
          ret = GLEW_ARB_texture_storage_multisample;
          continue;
        }
#endif
#ifdef GL_ARB_texture_swizzle
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_swizzle", 15))
        {
          ret = GLEW_ARB_texture_swizzle;
          continue;
        }
#endif
#ifdef GL_ARB_texture_view
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_view", 12))
        {
          ret = GLEW_ARB_texture_view;
          continue;
        }
#endif
#ifdef GL_ARB_timer_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"timer_query", 11))
        {
          ret = GLEW_ARB_timer_query;
          continue;
        }
#endif
#ifdef GL_ARB_transform_feedback2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transform_feedback2", 19))
        {
          ret = GLEW_ARB_transform_feedback2;
          continue;
        }
#endif
#ifdef GL_ARB_transform_feedback3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transform_feedback3", 19))
        {
          ret = GLEW_ARB_transform_feedback3;
          continue;
        }
#endif
#ifdef GL_ARB_transform_feedback_instanced
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transform_feedback_instanced", 28))
        {
          ret = GLEW_ARB_transform_feedback_instanced;
          continue;
        }
#endif
#ifdef GL_ARB_transform_feedback_overflow_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transform_feedback_overflow_query", 33))
        {
          ret = GLEW_ARB_transform_feedback_overflow_query;
          continue;
        }
#endif
#ifdef GL_ARB_transpose_matrix
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transpose_matrix", 16))
        {
          ret = GLEW_ARB_transpose_matrix;
          continue;
        }
#endif
#ifdef GL_ARB_uniform_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"uniform_buffer_object", 21))
        {
          ret = GLEW_ARB_uniform_buffer_object;
          continue;
        }
#endif
#ifdef GL_ARB_vertex_array_bgra
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_bgra", 17))
        {
          ret = GLEW_ARB_vertex_array_bgra;
          continue;
        }
#endif
#ifdef GL_ARB_vertex_array_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_object", 19))
        {
          ret = GLEW_ARB_vertex_array_object;
          continue;
        }
#endif
#ifdef GL_ARB_vertex_attrib_64bit
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_attrib_64bit", 19))
        {
          ret = GLEW_ARB_vertex_attrib_64bit;
          continue;
        }
#endif
#ifdef GL_ARB_vertex_attrib_binding
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_attrib_binding", 21))
        {
          ret = GLEW_ARB_vertex_attrib_binding;
          continue;
        }
#endif
#ifdef GL_ARB_vertex_blend
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_blend", 12))
        {
          ret = GLEW_ARB_vertex_blend;
          continue;
        }
#endif
#ifdef GL_ARB_vertex_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_buffer_object", 20))
        {
          ret = GLEW_ARB_vertex_buffer_object;
          continue;
        }
#endif
#ifdef GL_ARB_vertex_program
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_program", 14))
        {
          ret = GLEW_ARB_vertex_program;
          continue;
        }
#endif
#ifdef GL_ARB_vertex_shader
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_shader", 13))
        {
          ret = GLEW_ARB_vertex_shader;
          continue;
        }
#endif
#ifdef GL_ARB_vertex_type_10f_11f_11f_rev
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_type_10f_11f_11f_rev", 27))
        {
          ret = GLEW_ARB_vertex_type_10f_11f_11f_rev;
          continue;
        }
#endif
#ifdef GL_ARB_vertex_type_2_10_10_10_rev
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_type_2_10_10_10_rev", 26))
        {
          ret = GLEW_ARB_vertex_type_2_10_10_10_rev;
          continue;
        }
#endif
#ifdef GL_ARB_viewport_array
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"viewport_array", 14))
        {
          ret = GLEW_ARB_viewport_array;
          continue;
        }
#endif
#ifdef GL_ARB_window_pos
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"window_pos", 10))
        {
          ret = GLEW_ARB_window_pos;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"ATIX_", 5))
      {
#ifdef GL_ATIX_point_sprites
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"point_sprites", 13))
        {
          ret = GLEW_ATIX_point_sprites;
          continue;
        }
#endif
#ifdef GL_ATIX_texture_env_combine3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_combine3", 20))
        {
          ret = GLEW_ATIX_texture_env_combine3;
          continue;
        }
#endif
#ifdef GL_ATIX_texture_env_route
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_route", 17))
        {
          ret = GLEW_ATIX_texture_env_route;
          continue;
        }
#endif
#ifdef GL_ATIX_vertex_shader_output_point_size
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_shader_output_point_size", 31))
        {
          ret = GLEW_ATIX_vertex_shader_output_point_size;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"ATI_", 4))
      {
#ifdef GL_ATI_draw_buffers
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_buffers", 12))
        {
          ret = GLEW_ATI_draw_buffers;
          continue;
        }
#endif
#ifdef GL_ATI_element_array
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"element_array", 13))
        {
          ret = GLEW_ATI_element_array;
          continue;
        }
#endif
#ifdef GL_ATI_envmap_bumpmap
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"envmap_bumpmap", 14))
        {
          ret = GLEW_ATI_envmap_bumpmap;
          continue;
        }
#endif
#ifdef GL_ATI_fragment_shader
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_shader", 15))
        {
          ret = GLEW_ATI_fragment_shader;
          continue;
        }
#endif
#ifdef GL_ATI_map_object_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"map_object_buffer", 17))
        {
          ret = GLEW_ATI_map_object_buffer;
          continue;
        }
#endif
#ifdef GL_ATI_meminfo
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"meminfo", 7))
        {
          ret = GLEW_ATI_meminfo;
          continue;
        }
#endif
#ifdef GL_ATI_pn_triangles
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pn_triangles", 12))
        {
          ret = GLEW_ATI_pn_triangles;
          continue;
        }
#endif
#ifdef GL_ATI_separate_stencil
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"separate_stencil", 16))
        {
          ret = GLEW_ATI_separate_stencil;
          continue;
        }
#endif
#ifdef GL_ATI_shader_texture_lod
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_texture_lod", 18))
        {
          ret = GLEW_ATI_shader_texture_lod;
          continue;
        }
#endif
#ifdef GL_ATI_text_fragment_shader
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"text_fragment_shader", 20))
        {
          ret = GLEW_ATI_text_fragment_shader;
          continue;
        }
#endif
#ifdef GL_ATI_texture_compression_3dc
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_3dc", 23))
        {
          ret = GLEW_ATI_texture_compression_3dc;
          continue;
        }
#endif
#ifdef GL_ATI_texture_env_combine3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_combine3", 20))
        {
          ret = GLEW_ATI_texture_env_combine3;
          continue;
        }
#endif
#ifdef GL_ATI_texture_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_float", 13))
        {
          ret = GLEW_ATI_texture_float;
          continue;
        }
#endif
#ifdef GL_ATI_texture_mirror_once
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_mirror_once", 19))
        {
          ret = GLEW_ATI_texture_mirror_once;
          continue;
        }
#endif
#ifdef GL_ATI_vertex_array_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_object", 19))
        {
          ret = GLEW_ATI_vertex_array_object;
          continue;
        }
#endif
#ifdef GL_ATI_vertex_attrib_array_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_attrib_array_object", 26))
        {
          ret = GLEW_ATI_vertex_attrib_array_object;
          continue;
        }
#endif
#ifdef GL_ATI_vertex_streams
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_streams", 14))
        {
          ret = GLEW_ATI_vertex_streams;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"EXT_", 4))
      {
#ifdef GL_EXT_422_pixels
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"422_pixels", 10))
        {
          ret = GLEW_EXT_422_pixels;
          continue;
        }
#endif
#ifdef GL_EXT_Cg_shader
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"Cg_shader", 9))
        {
          ret = GLEW_EXT_Cg_shader;
          continue;
        }
#endif
#ifdef GL_EXT_abgr
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"abgr", 4))
        {
          ret = GLEW_EXT_abgr;
          continue;
        }
#endif
#ifdef GL_EXT_bgra
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"bgra", 4))
        {
          ret = GLEW_EXT_bgra;
          continue;
        }
#endif
#ifdef GL_EXT_bindable_uniform
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"bindable_uniform", 16))
        {
          ret = GLEW_EXT_bindable_uniform;
          continue;
        }
#endif
#ifdef GL_EXT_blend_color
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_color", 11))
        {
          ret = GLEW_EXT_blend_color;
          continue;
        }
#endif
#ifdef GL_EXT_blend_equation_separate
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_equation_separate", 23))
        {
          ret = GLEW_EXT_blend_equation_separate;
          continue;
        }
#endif
#ifdef GL_EXT_blend_func_separate
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_func_separate", 19))
        {
          ret = GLEW_EXT_blend_func_separate;
          continue;
        }
#endif
#ifdef GL_EXT_blend_logic_op
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_logic_op", 14))
        {
          ret = GLEW_EXT_blend_logic_op;
          continue;
        }
#endif
#ifdef GL_EXT_blend_minmax
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_minmax", 12))
        {
          ret = GLEW_EXT_blend_minmax;
          continue;
        }
#endif
#ifdef GL_EXT_blend_subtract
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_subtract", 14))
        {
          ret = GLEW_EXT_blend_subtract;
          continue;
        }
#endif
#ifdef GL_EXT_clip_volume_hint
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"clip_volume_hint", 16))
        {
          ret = GLEW_EXT_clip_volume_hint;
          continue;
        }
#endif
#ifdef GL_EXT_cmyka
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"cmyka", 5))
        {
          ret = GLEW_EXT_cmyka;
          continue;
        }
#endif
#ifdef GL_EXT_color_subtable
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"color_subtable", 14))
        {
          ret = GLEW_EXT_color_subtable;
          continue;
        }
#endif
#ifdef GL_EXT_compiled_vertex_array
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"compiled_vertex_array", 21))
        {
          ret = GLEW_EXT_compiled_vertex_array;
          continue;
        }
#endif
#ifdef GL_EXT_convolution
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"convolution", 11))
        {
          ret = GLEW_EXT_convolution;
          continue;
        }
#endif
#ifdef GL_EXT_coordinate_frame
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"coordinate_frame", 16))
        {
          ret = GLEW_EXT_coordinate_frame;
          continue;
        }
#endif
#ifdef GL_EXT_copy_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"copy_texture", 12))
        {
          ret = GLEW_EXT_copy_texture;
          continue;
        }
#endif
#ifdef GL_EXT_cull_vertex
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"cull_vertex", 11))
        {
          ret = GLEW_EXT_cull_vertex;
          continue;
        }
#endif
#ifdef GL_EXT_debug_label
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"debug_label", 11))
        {
          ret = GLEW_EXT_debug_label;
          continue;
        }
#endif
#ifdef GL_EXT_debug_marker
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"debug_marker", 12))
        {
          ret = GLEW_EXT_debug_marker;
          continue;
        }
#endif
#ifdef GL_EXT_depth_bounds_test
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_bounds_test", 17))
        {
          ret = GLEW_EXT_depth_bounds_test;
          continue;
        }
#endif
#ifdef GL_EXT_direct_state_access
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"direct_state_access", 19))
        {
          ret = GLEW_EXT_direct_state_access;
          continue;
        }
#endif
#ifdef GL_EXT_draw_buffers2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_buffers2", 13))
        {
          ret = GLEW_EXT_draw_buffers2;
          continue;
        }
#endif
#ifdef GL_EXT_draw_instanced
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_instanced", 14))
        {
          ret = GLEW_EXT_draw_instanced;
          continue;
        }
#endif
#ifdef GL_EXT_draw_range_elements
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_range_elements", 19))
        {
          ret = GLEW_EXT_draw_range_elements;
          continue;
        }
#endif
#ifdef GL_EXT_fog_coord
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fog_coord", 9))
        {
          ret = GLEW_EXT_fog_coord;
          continue;
        }
#endif
#ifdef GL_EXT_fragment_lighting
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_lighting", 17))
        {
          ret = GLEW_EXT_fragment_lighting;
          continue;
        }
#endif
#ifdef GL_EXT_framebuffer_blit
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_blit", 16))
        {
          ret = GLEW_EXT_framebuffer_blit;
          continue;
        }
#endif
#ifdef GL_EXT_framebuffer_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_multisample", 23))
        {
          ret = GLEW_EXT_framebuffer_multisample;
          continue;
        }
#endif
#ifdef GL_EXT_framebuffer_multisample_blit_scaled
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_multisample_blit_scaled", 35))
        {
          ret = GLEW_EXT_framebuffer_multisample_blit_scaled;
          continue;
        }
#endif
#ifdef GL_EXT_framebuffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_object", 18))
        {
          ret = GLEW_EXT_framebuffer_object;
          continue;
        }
#endif
#ifdef GL_EXT_framebuffer_sRGB
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_sRGB", 16))
        {
          ret = GLEW_EXT_framebuffer_sRGB;
          continue;
        }
#endif
#ifdef GL_EXT_geometry_shader4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"geometry_shader4", 16))
        {
          ret = GLEW_EXT_geometry_shader4;
          continue;
        }
#endif
#ifdef GL_EXT_gpu_program_parameters
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_program_parameters", 22))
        {
          ret = GLEW_EXT_gpu_program_parameters;
          continue;
        }
#endif
#ifdef GL_EXT_gpu_shader4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_shader4", 11))
        {
          ret = GLEW_EXT_gpu_shader4;
          continue;
        }
#endif
#ifdef GL_EXT_histogram
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"histogram", 9))
        {
          ret = GLEW_EXT_histogram;
          continue;
        }
#endif
#ifdef GL_EXT_index_array_formats
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"index_array_formats", 19))
        {
          ret = GLEW_EXT_index_array_formats;
          continue;
        }
#endif
#ifdef GL_EXT_index_func
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"index_func", 10))
        {
          ret = GLEW_EXT_index_func;
          continue;
        }
#endif
#ifdef GL_EXT_index_material
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"index_material", 14))
        {
          ret = GLEW_EXT_index_material;
          continue;
        }
#endif
#ifdef GL_EXT_index_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"index_texture", 13))
        {
          ret = GLEW_EXT_index_texture;
          continue;
        }
#endif
#ifdef GL_EXT_light_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"light_texture", 13))
        {
          ret = GLEW_EXT_light_texture;
          continue;
        }
#endif
#ifdef GL_EXT_misc_attribute
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"misc_attribute", 14))
        {
          ret = GLEW_EXT_misc_attribute;
          continue;
        }
#endif
#ifdef GL_EXT_multi_draw_arrays
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multi_draw_arrays", 17))
        {
          ret = GLEW_EXT_multi_draw_arrays;
          continue;
        }
#endif
#ifdef GL_EXT_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample", 11))
        {
          ret = GLEW_EXT_multisample;
          continue;
        }
#endif
#ifdef GL_EXT_packed_depth_stencil
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"packed_depth_stencil", 20))
        {
          ret = GLEW_EXT_packed_depth_stencil;
          continue;
        }
#endif
#ifdef GL_EXT_packed_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"packed_float", 12))
        {
          ret = GLEW_EXT_packed_float;
          continue;
        }
#endif
#ifdef GL_EXT_packed_pixels
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"packed_pixels", 13))
        {
          ret = GLEW_EXT_packed_pixels;
          continue;
        }
#endif
#ifdef GL_EXT_paletted_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"paletted_texture", 16))
        {
          ret = GLEW_EXT_paletted_texture;
          continue;
        }
#endif
#ifdef GL_EXT_pixel_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_buffer_object", 19))
        {
          ret = GLEW_EXT_pixel_buffer_object;
          continue;
        }
#endif
#ifdef GL_EXT_pixel_transform
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_transform", 15))
        {
          ret = GLEW_EXT_pixel_transform;
          continue;
        }
#endif
#ifdef GL_EXT_pixel_transform_color_table
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_transform_color_table", 27))
        {
          ret = GLEW_EXT_pixel_transform_color_table;
          continue;
        }
#endif
#ifdef GL_EXT_point_parameters
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"point_parameters", 16))
        {
          ret = GLEW_EXT_point_parameters;
          continue;
        }
#endif
#ifdef GL_EXT_polygon_offset
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"polygon_offset", 14))
        {
          ret = GLEW_EXT_polygon_offset;
          continue;
        }
#endif
#ifdef GL_EXT_polygon_offset_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"polygon_offset_clamp", 20))
        {
          ret = GLEW_EXT_polygon_offset_clamp;
          continue;
        }
#endif
#ifdef GL_EXT_post_depth_coverage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"post_depth_coverage", 19))
        {
          ret = GLEW_EXT_post_depth_coverage;
          continue;
        }
#endif
#ifdef GL_EXT_provoking_vertex
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"provoking_vertex", 16))
        {
          ret = GLEW_EXT_provoking_vertex;
          continue;
        }
#endif
#ifdef GL_EXT_raster_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"raster_multisample", 18))
        {
          ret = GLEW_EXT_raster_multisample;
          continue;
        }
#endif
#ifdef GL_EXT_rescale_normal
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"rescale_normal", 14))
        {
          ret = GLEW_EXT_rescale_normal;
          continue;
        }
#endif
#ifdef GL_EXT_scene_marker
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"scene_marker", 12))
        {
          ret = GLEW_EXT_scene_marker;
          continue;
        }
#endif
#ifdef GL_EXT_secondary_color
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"secondary_color", 15))
        {
          ret = GLEW_EXT_secondary_color;
          continue;
        }
#endif
#ifdef GL_EXT_separate_shader_objects
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"separate_shader_objects", 23))
        {
          ret = GLEW_EXT_separate_shader_objects;
          continue;
        }
#endif
#ifdef GL_EXT_separate_specular_color
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"separate_specular_color", 23))
        {
          ret = GLEW_EXT_separate_specular_color;
          continue;
        }
#endif
#ifdef GL_EXT_shader_image_load_formatted
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_image_load_formatted", 27))
        {
          ret = GLEW_EXT_shader_image_load_formatted;
          continue;
        }
#endif
#ifdef GL_EXT_shader_image_load_store
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_image_load_store", 23))
        {
          ret = GLEW_EXT_shader_image_load_store;
          continue;
        }
#endif
#ifdef GL_EXT_shader_integer_mix
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_integer_mix", 18))
        {
          ret = GLEW_EXT_shader_integer_mix;
          continue;
        }
#endif
#ifdef GL_EXT_shadow_funcs
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shadow_funcs", 12))
        {
          ret = GLEW_EXT_shadow_funcs;
          continue;
        }
#endif
#ifdef GL_EXT_shared_texture_palette
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shared_texture_palette", 22))
        {
          ret = GLEW_EXT_shared_texture_palette;
          continue;
        }
#endif
#ifdef GL_EXT_sparse_texture2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sparse_texture2", 15))
        {
          ret = GLEW_EXT_sparse_texture2;
          continue;
        }
#endif
#ifdef GL_EXT_stencil_clear_tag
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"stencil_clear_tag", 17))
        {
          ret = GLEW_EXT_stencil_clear_tag;
          continue;
        }
#endif
#ifdef GL_EXT_stencil_two_side
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"stencil_two_side", 16))
        {
          ret = GLEW_EXT_stencil_two_side;
          continue;
        }
#endif
#ifdef GL_EXT_stencil_wrap
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"stencil_wrap", 12))
        {
          ret = GLEW_EXT_stencil_wrap;
          continue;
        }
#endif
#ifdef GL_EXT_subtexture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"subtexture", 10))
        {
          ret = GLEW_EXT_subtexture;
          continue;
        }
#endif
#ifdef GL_EXT_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture", 7))
        {
          ret = GLEW_EXT_texture;
          continue;
        }
#endif
#ifdef GL_EXT_texture3D
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture3D", 9))
        {
          ret = GLEW_EXT_texture3D;
          continue;
        }
#endif
#ifdef GL_EXT_texture_array
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_array", 13))
        {
          ret = GLEW_EXT_texture_array;
          continue;
        }
#endif
#ifdef GL_EXT_texture_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_buffer_object", 21))
        {
          ret = GLEW_EXT_texture_buffer_object;
          continue;
        }
#endif
#ifdef GL_EXT_texture_compression_dxt1
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_dxt1", 24))
        {
          ret = GLEW_EXT_texture_compression_dxt1;
          continue;
        }
#endif
#ifdef GL_EXT_texture_compression_latc
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_latc", 24))
        {
          ret = GLEW_EXT_texture_compression_latc;
          continue;
        }
#endif
#ifdef GL_EXT_texture_compression_rgtc
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_rgtc", 24))
        {
          ret = GLEW_EXT_texture_compression_rgtc;
          continue;
        }
#endif
#ifdef GL_EXT_texture_compression_s3tc
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_s3tc", 24))
        {
          ret = GLEW_EXT_texture_compression_s3tc;
          continue;
        }
#endif
#ifdef GL_EXT_texture_cube_map
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_cube_map", 16))
        {
          ret = GLEW_EXT_texture_cube_map;
          continue;
        }
#endif
#ifdef GL_EXT_texture_edge_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_edge_clamp", 18))
        {
          ret = GLEW_EXT_texture_edge_clamp;
          continue;
        }
#endif
#ifdef GL_EXT_texture_env
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env", 11))
        {
          ret = GLEW_EXT_texture_env;
          continue;
        }
#endif
#ifdef GL_EXT_texture_env_add
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_add", 15))
        {
          ret = GLEW_EXT_texture_env_add;
          continue;
        }
#endif
#ifdef GL_EXT_texture_env_combine
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_combine", 19))
        {
          ret = GLEW_EXT_texture_env_combine;
          continue;
        }
#endif
#ifdef GL_EXT_texture_env_dot3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_dot3", 16))
        {
          ret = GLEW_EXT_texture_env_dot3;
          continue;
        }
#endif
#ifdef GL_EXT_texture_filter_anisotropic
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_filter_anisotropic", 26))
        {
          ret = GLEW_EXT_texture_filter_anisotropic;
          continue;
        }
#endif
#ifdef GL_EXT_texture_filter_minmax
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_filter_minmax", 21))
        {
          ret = GLEW_EXT_texture_filter_minmax;
          continue;
        }
#endif
#ifdef GL_EXT_texture_integer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_integer", 15))
        {
          ret = GLEW_EXT_texture_integer;
          continue;
        }
#endif
#ifdef GL_EXT_texture_lod_bias
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_lod_bias", 16))
        {
          ret = GLEW_EXT_texture_lod_bias;
          continue;
        }
#endif
#ifdef GL_EXT_texture_mirror_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_mirror_clamp", 20))
        {
          ret = GLEW_EXT_texture_mirror_clamp;
          continue;
        }
#endif
#ifdef GL_EXT_texture_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_object", 14))
        {
          ret = GLEW_EXT_texture_object;
          continue;
        }
#endif
#ifdef GL_EXT_texture_perturb_normal
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_perturb_normal", 22))
        {
          ret = GLEW_EXT_texture_perturb_normal;
          continue;
        }
#endif
#ifdef GL_EXT_texture_rectangle
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_rectangle", 17))
        {
          ret = GLEW_EXT_texture_rectangle;
          continue;
        }
#endif
#ifdef GL_EXT_texture_sRGB
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_sRGB", 12))
        {
          ret = GLEW_EXT_texture_sRGB;
          continue;
        }
#endif
#ifdef GL_EXT_texture_sRGB_decode
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_sRGB_decode", 19))
        {
          ret = GLEW_EXT_texture_sRGB_decode;
          continue;
        }
#endif
#ifdef GL_EXT_texture_shared_exponent
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_shared_exponent", 23))
        {
          ret = GLEW_EXT_texture_shared_exponent;
          continue;
        }
#endif
#ifdef GL_EXT_texture_snorm
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_snorm", 13))
        {
          ret = GLEW_EXT_texture_snorm;
          continue;
        }
#endif
#ifdef GL_EXT_texture_swizzle
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_swizzle", 15))
        {
          ret = GLEW_EXT_texture_swizzle;
          continue;
        }
#endif
#ifdef GL_EXT_timer_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"timer_query", 11))
        {
          ret = GLEW_EXT_timer_query;
          continue;
        }
#endif
#ifdef GL_EXT_transform_feedback
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transform_feedback", 18))
        {
          ret = GLEW_EXT_transform_feedback;
          continue;
        }
#endif
#ifdef GL_EXT_vertex_array
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array", 12))
        {
          ret = GLEW_EXT_vertex_array;
          continue;
        }
#endif
#ifdef GL_EXT_vertex_array_bgra
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_bgra", 17))
        {
          ret = GLEW_EXT_vertex_array_bgra;
          continue;
        }
#endif
#ifdef GL_EXT_vertex_attrib_64bit
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_attrib_64bit", 19))
        {
          ret = GLEW_EXT_vertex_attrib_64bit;
          continue;
        }
#endif
#ifdef GL_EXT_vertex_shader
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_shader", 13))
        {
          ret = GLEW_EXT_vertex_shader;
          continue;
        }
#endif
#ifdef GL_EXT_vertex_weighting
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_weighting", 16))
        {
          ret = GLEW_EXT_vertex_weighting;
          continue;
        }
#endif
#ifdef GL_EXT_x11_sync_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"x11_sync_object", 15))
        {
          ret = GLEW_EXT_x11_sync_object;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"GREMEDY_", 8))
      {
#ifdef GL_GREMEDY_frame_terminator
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"frame_terminator", 16))
        {
          ret = GLEW_GREMEDY_frame_terminator;
          continue;
        }
#endif
#ifdef GL_GREMEDY_string_marker
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"string_marker", 13))
        {
          ret = GLEW_GREMEDY_string_marker;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"HP_", 3))
      {
#ifdef GL_HP_convolution_border_modes
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"convolution_border_modes", 24))
        {
          ret = GLEW_HP_convolution_border_modes;
          continue;
        }
#endif
#ifdef GL_HP_image_transform
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"image_transform", 15))
        {
          ret = GLEW_HP_image_transform;
          continue;
        }
#endif
#ifdef GL_HP_occlusion_test
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"occlusion_test", 14))
        {
          ret = GLEW_HP_occlusion_test;
          continue;
        }
#endif
#ifdef GL_HP_texture_lighting
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_lighting", 16))
        {
          ret = GLEW_HP_texture_lighting;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"IBM_", 4))
      {
#ifdef GL_IBM_cull_vertex
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"cull_vertex", 11))
        {
          ret = GLEW_IBM_cull_vertex;
          continue;
        }
#endif
#ifdef GL_IBM_multimode_draw_arrays
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multimode_draw_arrays", 21))
        {
          ret = GLEW_IBM_multimode_draw_arrays;
          continue;
        }
#endif
#ifdef GL_IBM_rasterpos_clip
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"rasterpos_clip", 14))
        {
          ret = GLEW_IBM_rasterpos_clip;
          continue;
        }
#endif
#ifdef GL_IBM_static_data
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"static_data", 11))
        {
          ret = GLEW_IBM_static_data;
          continue;
        }
#endif
#ifdef GL_IBM_texture_mirrored_repeat
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_mirrored_repeat", 23))
        {
          ret = GLEW_IBM_texture_mirrored_repeat;
          continue;
        }
#endif
#ifdef GL_IBM_vertex_array_lists
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_lists", 18))
        {
          ret = GLEW_IBM_vertex_array_lists;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"INGR_", 5))
      {
#ifdef GL_INGR_color_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"color_clamp", 11))
        {
          ret = GLEW_INGR_color_clamp;
          continue;
        }
#endif
#ifdef GL_INGR_interlace_read
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"interlace_read", 14))
        {
          ret = GLEW_INGR_interlace_read;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"INTEL_", 6))
      {
#ifdef GL_INTEL_fragment_shader_ordering
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_shader_ordering", 24))
        {
          ret = GLEW_INTEL_fragment_shader_ordering;
          continue;
        }
#endif
#ifdef GL_INTEL_framebuffer_CMAA
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_CMAA", 16))
        {
          ret = GLEW_INTEL_framebuffer_CMAA;
          continue;
        }
#endif
#ifdef GL_INTEL_map_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"map_texture", 11))
        {
          ret = GLEW_INTEL_map_texture;
          continue;
        }
#endif
#ifdef GL_INTEL_parallel_arrays
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"parallel_arrays", 15))
        {
          ret = GLEW_INTEL_parallel_arrays;
          continue;
        }
#endif
#ifdef GL_INTEL_performance_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"performance_query", 17))
        {
          ret = GLEW_INTEL_performance_query;
          continue;
        }
#endif
#ifdef GL_INTEL_texture_scissor
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_scissor", 15))
        {
          ret = GLEW_INTEL_texture_scissor;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"KHR_", 4))
      {
#ifdef GL_KHR_blend_equation_advanced
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_equation_advanced", 23))
        {
          ret = GLEW_KHR_blend_equation_advanced;
          continue;
        }
#endif
#ifdef GL_KHR_blend_equation_advanced_coherent
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_equation_advanced_coherent", 32))
        {
          ret = GLEW_KHR_blend_equation_advanced_coherent;
          continue;
        }
#endif
#ifdef GL_KHR_context_flush_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"context_flush_control", 21))
        {
          ret = GLEW_KHR_context_flush_control;
          continue;
        }
#endif
#ifdef GL_KHR_debug
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"debug", 5))
        {
          ret = GLEW_KHR_debug;
          continue;
        }
#endif
#ifdef GL_KHR_no_error
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"no_error", 8))
        {
          ret = GLEW_KHR_no_error;
          continue;
        }
#endif
#ifdef GL_KHR_robust_buffer_access_behavior
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"robust_buffer_access_behavior", 29))
        {
          ret = GLEW_KHR_robust_buffer_access_behavior;
          continue;
        }
#endif
#ifdef GL_KHR_robustness
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"robustness", 10))
        {
          ret = GLEW_KHR_robustness;
          continue;
        }
#endif
#ifdef GL_KHR_texture_compression_astc_hdr
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_astc_hdr", 28))
        {
          ret = GLEW_KHR_texture_compression_astc_hdr;
          continue;
        }
#endif
#ifdef GL_KHR_texture_compression_astc_ldr
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_astc_ldr", 28))
        {
          ret = GLEW_KHR_texture_compression_astc_ldr;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"KTX_", 4))
      {
#ifdef GL_KTX_buffer_region
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"buffer_region", 13))
        {
          ret = GLEW_KTX_buffer_region;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"MESAX_", 6))
      {
#ifdef GL_MESAX_texture_stack
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_stack", 13))
        {
          ret = GLEW_MESAX_texture_stack;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"MESA_", 5))
      {
#ifdef GL_MESA_pack_invert
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pack_invert", 11))
        {
          ret = GLEW_MESA_pack_invert;
          continue;
        }
#endif
#ifdef GL_MESA_resize_buffers
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"resize_buffers", 14))
        {
          ret = GLEW_MESA_resize_buffers;
          continue;
        }
#endif
#ifdef GL_MESA_window_pos
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"window_pos", 10))
        {
          ret = GLEW_MESA_window_pos;
          continue;
        }
#endif
#ifdef GL_MESA_ycbcr_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"ycbcr_texture", 13))
        {
          ret = GLEW_MESA_ycbcr_texture;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"NVX_", 4))
      {
#ifdef GL_NVX_conditional_render
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"conditional_render", 18))
        {
          ret = GLEW_NVX_conditional_render;
          continue;
        }
#endif
#ifdef GL_NVX_gpu_memory_info
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_memory_info", 15))
        {
          ret = GLEW_NVX_gpu_memory_info;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"NV_", 3))
      {
#ifdef GL_NV_bindless_multi_draw_indirect
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"bindless_multi_draw_indirect", 28))
        {
          ret = GLEW_NV_bindless_multi_draw_indirect;
          continue;
        }
#endif
#ifdef GL_NV_bindless_multi_draw_indirect_count
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"bindless_multi_draw_indirect_count", 34))
        {
          ret = GLEW_NV_bindless_multi_draw_indirect_count;
          continue;
        }
#endif
#ifdef GL_NV_bindless_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"bindless_texture", 16))
        {
          ret = GLEW_NV_bindless_texture;
          continue;
        }
#endif
#ifdef GL_NV_blend_equation_advanced
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_equation_advanced", 23))
        {
          ret = GLEW_NV_blend_equation_advanced;
          continue;
        }
#endif
#ifdef GL_NV_blend_equation_advanced_coherent
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_equation_advanced_coherent", 32))
        {
          ret = GLEW_NV_blend_equation_advanced_coherent;
          continue;
        }
#endif
#ifdef GL_NV_blend_square
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_square", 12))
        {
          ret = GLEW_NV_blend_square;
          continue;
        }
#endif
#ifdef GL_NV_compute_program5
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"compute_program5", 16))
        {
          ret = GLEW_NV_compute_program5;
          continue;
        }
#endif
#ifdef GL_NV_conditional_render
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"conditional_render", 18))
        {
          ret = GLEW_NV_conditional_render;
          continue;
        }
#endif
#ifdef GL_NV_conservative_raster
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"conservative_raster", 19))
        {
          ret = GLEW_NV_conservative_raster;
          continue;
        }
#endif
#ifdef GL_NV_conservative_raster_dilate
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"conservative_raster_dilate", 26))
        {
          ret = GLEW_NV_conservative_raster_dilate;
          continue;
        }
#endif
#ifdef GL_NV_copy_depth_to_color
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"copy_depth_to_color", 19))
        {
          ret = GLEW_NV_copy_depth_to_color;
          continue;
        }
#endif
#ifdef GL_NV_copy_image
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"copy_image", 10))
        {
          ret = GLEW_NV_copy_image;
          continue;
        }
#endif
#ifdef GL_NV_deep_texture3D
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"deep_texture3D", 14))
        {
          ret = GLEW_NV_deep_texture3D;
          continue;
        }
#endif
#ifdef GL_NV_depth_buffer_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_buffer_float", 18))
        {
          ret = GLEW_NV_depth_buffer_float;
          continue;
        }
#endif
#ifdef GL_NV_depth_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_clamp", 11))
        {
          ret = GLEW_NV_depth_clamp;
          continue;
        }
#endif
#ifdef GL_NV_depth_range_unclamped
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_range_unclamped", 21))
        {
          ret = GLEW_NV_depth_range_unclamped;
          continue;
        }
#endif
#ifdef GL_NV_draw_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"draw_texture", 12))
        {
          ret = GLEW_NV_draw_texture;
          continue;
        }
#endif
#ifdef GL_NV_evaluators
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"evaluators", 10))
        {
          ret = GLEW_NV_evaluators;
          continue;
        }
#endif
#ifdef GL_NV_explicit_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"explicit_multisample", 20))
        {
          ret = GLEW_NV_explicit_multisample;
          continue;
        }
#endif
#ifdef GL_NV_fence
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fence", 5))
        {
          ret = GLEW_NV_fence;
          continue;
        }
#endif
#ifdef GL_NV_fill_rectangle
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fill_rectangle", 14))
        {
          ret = GLEW_NV_fill_rectangle;
          continue;
        }
#endif
#ifdef GL_NV_float_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"float_buffer", 12))
        {
          ret = GLEW_NV_float_buffer;
          continue;
        }
#endif
#ifdef GL_NV_fog_distance
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fog_distance", 12))
        {
          ret = GLEW_NV_fog_distance;
          continue;
        }
#endif
#ifdef GL_NV_fragment_coverage_to_color
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_coverage_to_color", 26))
        {
          ret = GLEW_NV_fragment_coverage_to_color;
          continue;
        }
#endif
#ifdef GL_NV_fragment_program
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_program", 16))
        {
          ret = GLEW_NV_fragment_program;
          continue;
        }
#endif
#ifdef GL_NV_fragment_program2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_program2", 17))
        {
          ret = GLEW_NV_fragment_program2;
          continue;
        }
#endif
#ifdef GL_NV_fragment_program4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_program4", 17))
        {
          ret = GLEW_NV_fragment_program4;
          continue;
        }
#endif
#ifdef GL_NV_fragment_program_option
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_program_option", 23))
        {
          ret = GLEW_NV_fragment_program_option;
          continue;
        }
#endif
#ifdef GL_NV_fragment_shader_interlock
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_shader_interlock", 25))
        {
          ret = GLEW_NV_fragment_shader_interlock;
          continue;
        }
#endif
#ifdef GL_NV_framebuffer_mixed_samples
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_mixed_samples", 25))
        {
          ret = GLEW_NV_framebuffer_mixed_samples;
          continue;
        }
#endif
#ifdef GL_NV_framebuffer_multisample_coverage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_multisample_coverage", 32))
        {
          ret = GLEW_NV_framebuffer_multisample_coverage;
          continue;
        }
#endif
#ifdef GL_NV_geometry_program4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"geometry_program4", 17))
        {
          ret = GLEW_NV_geometry_program4;
          continue;
        }
#endif
#ifdef GL_NV_geometry_shader4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"geometry_shader4", 16))
        {
          ret = GLEW_NV_geometry_shader4;
          continue;
        }
#endif
#ifdef GL_NV_geometry_shader_passthrough
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"geometry_shader_passthrough", 27))
        {
          ret = GLEW_NV_geometry_shader_passthrough;
          continue;
        }
#endif
#ifdef GL_NV_gpu_program4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_program4", 12))
        {
          ret = GLEW_NV_gpu_program4;
          continue;
        }
#endif
#ifdef GL_NV_gpu_program5
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_program5", 12))
        {
          ret = GLEW_NV_gpu_program5;
          continue;
        }
#endif
#ifdef GL_NV_gpu_program5_mem_extended
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_program5_mem_extended", 25))
        {
          ret = GLEW_NV_gpu_program5_mem_extended;
          continue;
        }
#endif
#ifdef GL_NV_gpu_program_fp64
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_program_fp64", 16))
        {
          ret = GLEW_NV_gpu_program_fp64;
          continue;
        }
#endif
#ifdef GL_NV_gpu_shader5
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_shader5", 11))
        {
          ret = GLEW_NV_gpu_shader5;
          continue;
        }
#endif
#ifdef GL_NV_half_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"half_float", 10))
        {
          ret = GLEW_NV_half_float;
          continue;
        }
#endif
#ifdef GL_NV_internalformat_sample_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"internalformat_sample_query", 27))
        {
          ret = GLEW_NV_internalformat_sample_query;
          continue;
        }
#endif
#ifdef GL_NV_light_max_exponent
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"light_max_exponent", 18))
        {
          ret = GLEW_NV_light_max_exponent;
          continue;
        }
#endif
#ifdef GL_NV_multisample_coverage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample_coverage", 20))
        {
          ret = GLEW_NV_multisample_coverage;
          continue;
        }
#endif
#ifdef GL_NV_multisample_filter_hint
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample_filter_hint", 23))
        {
          ret = GLEW_NV_multisample_filter_hint;
          continue;
        }
#endif
#ifdef GL_NV_occlusion_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"occlusion_query", 15))
        {
          ret = GLEW_NV_occlusion_query;
          continue;
        }
#endif
#ifdef GL_NV_packed_depth_stencil
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"packed_depth_stencil", 20))
        {
          ret = GLEW_NV_packed_depth_stencil;
          continue;
        }
#endif
#ifdef GL_NV_parameter_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"parameter_buffer_object", 23))
        {
          ret = GLEW_NV_parameter_buffer_object;
          continue;
        }
#endif
#ifdef GL_NV_parameter_buffer_object2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"parameter_buffer_object2", 24))
        {
          ret = GLEW_NV_parameter_buffer_object2;
          continue;
        }
#endif
#ifdef GL_NV_path_rendering
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"path_rendering", 14))
        {
          ret = GLEW_NV_path_rendering;
          continue;
        }
#endif
#ifdef GL_NV_path_rendering_shared_edge
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"path_rendering_shared_edge", 26))
        {
          ret = GLEW_NV_path_rendering_shared_edge;
          continue;
        }
#endif
#ifdef GL_NV_pixel_data_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_data_range", 16))
        {
          ret = GLEW_NV_pixel_data_range;
          continue;
        }
#endif
#ifdef GL_NV_point_sprite
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"point_sprite", 12))
        {
          ret = GLEW_NV_point_sprite;
          continue;
        }
#endif
#ifdef GL_NV_present_video
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"present_video", 13))
        {
          ret = GLEW_NV_present_video;
          continue;
        }
#endif
#ifdef GL_NV_primitive_restart
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"primitive_restart", 17))
        {
          ret = GLEW_NV_primitive_restart;
          continue;
        }
#endif
#ifdef GL_NV_register_combiners
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"register_combiners", 18))
        {
          ret = GLEW_NV_register_combiners;
          continue;
        }
#endif
#ifdef GL_NV_register_combiners2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"register_combiners2", 19))
        {
          ret = GLEW_NV_register_combiners2;
          continue;
        }
#endif
#ifdef GL_NV_sample_locations
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sample_locations", 16))
        {
          ret = GLEW_NV_sample_locations;
          continue;
        }
#endif
#ifdef GL_NV_sample_mask_override_coverage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sample_mask_override_coverage", 29))
        {
          ret = GLEW_NV_sample_mask_override_coverage;
          continue;
        }
#endif
#ifdef GL_NV_shader_atomic_counters
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_atomic_counters", 22))
        {
          ret = GLEW_NV_shader_atomic_counters;
          continue;
        }
#endif
#ifdef GL_NV_shader_atomic_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_atomic_float", 19))
        {
          ret = GLEW_NV_shader_atomic_float;
          continue;
        }
#endif
#ifdef GL_NV_shader_atomic_fp16_vector
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_atomic_fp16_vector", 25))
        {
          ret = GLEW_NV_shader_atomic_fp16_vector;
          continue;
        }
#endif
#ifdef GL_NV_shader_atomic_int64
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_atomic_int64", 19))
        {
          ret = GLEW_NV_shader_atomic_int64;
          continue;
        }
#endif
#ifdef GL_NV_shader_buffer_load
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_buffer_load", 18))
        {
          ret = GLEW_NV_shader_buffer_load;
          continue;
        }
#endif
#ifdef GL_NV_shader_storage_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_storage_buffer_object", 28))
        {
          ret = GLEW_NV_shader_storage_buffer_object;
          continue;
        }
#endif
#ifdef GL_NV_shader_thread_group
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_thread_group", 19))
        {
          ret = GLEW_NV_shader_thread_group;
          continue;
        }
#endif
#ifdef GL_NV_shader_thread_shuffle
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shader_thread_shuffle", 21))
        {
          ret = GLEW_NV_shader_thread_shuffle;
          continue;
        }
#endif
#ifdef GL_NV_tessellation_program5
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"tessellation_program5", 21))
        {
          ret = GLEW_NV_tessellation_program5;
          continue;
        }
#endif
#ifdef GL_NV_texgen_emboss
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texgen_emboss", 13))
        {
          ret = GLEW_NV_texgen_emboss;
          continue;
        }
#endif
#ifdef GL_NV_texgen_reflection
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texgen_reflection", 17))
        {
          ret = GLEW_NV_texgen_reflection;
          continue;
        }
#endif
#ifdef GL_NV_texture_barrier
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_barrier", 15))
        {
          ret = GLEW_NV_texture_barrier;
          continue;
        }
#endif
#ifdef GL_NV_texture_compression_vtc
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_compression_vtc", 23))
        {
          ret = GLEW_NV_texture_compression_vtc;
          continue;
        }
#endif
#ifdef GL_NV_texture_env_combine4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_env_combine4", 20))
        {
          ret = GLEW_NV_texture_env_combine4;
          continue;
        }
#endif
#ifdef GL_NV_texture_expand_normal
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_expand_normal", 21))
        {
          ret = GLEW_NV_texture_expand_normal;
          continue;
        }
#endif
#ifdef GL_NV_texture_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_multisample", 19))
        {
          ret = GLEW_NV_texture_multisample;
          continue;
        }
#endif
#ifdef GL_NV_texture_rectangle
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_rectangle", 17))
        {
          ret = GLEW_NV_texture_rectangle;
          continue;
        }
#endif
#ifdef GL_NV_texture_shader
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_shader", 14))
        {
          ret = GLEW_NV_texture_shader;
          continue;
        }
#endif
#ifdef GL_NV_texture_shader2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_shader2", 15))
        {
          ret = GLEW_NV_texture_shader2;
          continue;
        }
#endif
#ifdef GL_NV_texture_shader3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_shader3", 15))
        {
          ret = GLEW_NV_texture_shader3;
          continue;
        }
#endif
#ifdef GL_NV_transform_feedback
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transform_feedback", 18))
        {
          ret = GLEW_NV_transform_feedback;
          continue;
        }
#endif
#ifdef GL_NV_transform_feedback2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"transform_feedback2", 19))
        {
          ret = GLEW_NV_transform_feedback2;
          continue;
        }
#endif
#ifdef GL_NV_uniform_buffer_unified_memory
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"uniform_buffer_unified_memory", 29))
        {
          ret = GLEW_NV_uniform_buffer_unified_memory;
          continue;
        }
#endif
#ifdef GL_NV_vdpau_interop
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vdpau_interop", 13))
        {
          ret = GLEW_NV_vdpau_interop;
          continue;
        }
#endif
#ifdef GL_NV_vertex_array_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_range", 18))
        {
          ret = GLEW_NV_vertex_array_range;
          continue;
        }
#endif
#ifdef GL_NV_vertex_array_range2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_range2", 19))
        {
          ret = GLEW_NV_vertex_array_range2;
          continue;
        }
#endif
#ifdef GL_NV_vertex_attrib_integer_64bit
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_attrib_integer_64bit", 27))
        {
          ret = GLEW_NV_vertex_attrib_integer_64bit;
          continue;
        }
#endif
#ifdef GL_NV_vertex_buffer_unified_memory
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_buffer_unified_memory", 28))
        {
          ret = GLEW_NV_vertex_buffer_unified_memory;
          continue;
        }
#endif
#ifdef GL_NV_vertex_program
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_program", 14))
        {
          ret = GLEW_NV_vertex_program;
          continue;
        }
#endif
#ifdef GL_NV_vertex_program1_1
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_program1_1", 17))
        {
          ret = GLEW_NV_vertex_program1_1;
          continue;
        }
#endif
#ifdef GL_NV_vertex_program2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_program2", 15))
        {
          ret = GLEW_NV_vertex_program2;
          continue;
        }
#endif
#ifdef GL_NV_vertex_program2_option
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_program2_option", 22))
        {
          ret = GLEW_NV_vertex_program2_option;
          continue;
        }
#endif
#ifdef GL_NV_vertex_program3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_program3", 15))
        {
          ret = GLEW_NV_vertex_program3;
          continue;
        }
#endif
#ifdef GL_NV_vertex_program4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_program4", 15))
        {
          ret = GLEW_NV_vertex_program4;
          continue;
        }
#endif
#ifdef GL_NV_video_capture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"video_capture", 13))
        {
          ret = GLEW_NV_video_capture;
          continue;
        }
#endif
#ifdef GL_NV_viewport_array2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"viewport_array2", 15))
        {
          ret = GLEW_NV_viewport_array2;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"OES_", 4))
      {
#ifdef GL_OES_byte_coordinates
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"byte_coordinates", 16))
        {
          ret = GLEW_OES_byte_coordinates;
          continue;
        }
#endif
#ifdef GL_OES_compressed_paletted_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"compressed_paletted_texture", 27))
        {
          ret = GLEW_OES_compressed_paletted_texture;
          continue;
        }
#endif
#ifdef GL_OES_read_format
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"read_format", 11))
        {
          ret = GLEW_OES_read_format;
          continue;
        }
#endif
#ifdef GL_OES_single_precision
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"single_precision", 16))
        {
          ret = GLEW_OES_single_precision;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"OML_", 4))
      {
#ifdef GL_OML_interlace
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"interlace", 9))
        {
          ret = GLEW_OML_interlace;
          continue;
        }
#endif
#ifdef GL_OML_resample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"resample", 8))
        {
          ret = GLEW_OML_resample;
          continue;
        }
#endif
#ifdef GL_OML_subsample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"subsample", 9))
        {
          ret = GLEW_OML_subsample;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"OVR_", 4))
      {
#ifdef GL_OVR_multiview
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multiview", 9))
        {
          ret = GLEW_OVR_multiview;
          continue;
        }
#endif
#ifdef GL_OVR_multiview2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multiview2", 10))
        {
          ret = GLEW_OVR_multiview2;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"PGI_", 4))
      {
#ifdef GL_PGI_misc_hints
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"misc_hints", 10))
        {
          ret = GLEW_PGI_misc_hints;
          continue;
        }
#endif
#ifdef GL_PGI_vertex_hints
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_hints", 12))
        {
          ret = GLEW_PGI_vertex_hints;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"REGAL_", 6))
      {
#ifdef GL_REGAL_ES1_0_compatibility
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"ES1_0_compatibility", 19))
        {
          ret = GLEW_REGAL_ES1_0_compatibility;
          continue;
        }
#endif
#ifdef GL_REGAL_ES1_1_compatibility
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"ES1_1_compatibility", 19))
        {
          ret = GLEW_REGAL_ES1_1_compatibility;
          continue;
        }
#endif
#ifdef GL_REGAL_enable
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"enable", 6))
        {
          ret = GLEW_REGAL_enable;
          continue;
        }
#endif
#ifdef GL_REGAL_error_string
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"error_string", 12))
        {
          ret = GLEW_REGAL_error_string;
          continue;
        }
#endif
#ifdef GL_REGAL_extension_query
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"extension_query", 15))
        {
          ret = GLEW_REGAL_extension_query;
          continue;
        }
#endif
#ifdef GL_REGAL_log
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"log", 3))
        {
          ret = GLEW_REGAL_log;
          continue;
        }
#endif
#ifdef GL_REGAL_proc_address
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"proc_address", 12))
        {
          ret = GLEW_REGAL_proc_address;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"REND_", 5))
      {
#ifdef GL_REND_screen_coordinates
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"screen_coordinates", 18))
        {
          ret = GLEW_REND_screen_coordinates;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"S3_", 3))
      {
#ifdef GL_S3_s3tc
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"s3tc", 4))
        {
          ret = GLEW_S3_s3tc;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"SGIS_", 5))
      {
#ifdef GL_SGIS_color_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"color_range", 11))
        {
          ret = GLEW_SGIS_color_range;
          continue;
        }
#endif
#ifdef GL_SGIS_detail_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"detail_texture", 14))
        {
          ret = GLEW_SGIS_detail_texture;
          continue;
        }
#endif
#ifdef GL_SGIS_fog_function
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fog_function", 12))
        {
          ret = GLEW_SGIS_fog_function;
          continue;
        }
#endif
#ifdef GL_SGIS_generate_mipmap
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"generate_mipmap", 15))
        {
          ret = GLEW_SGIS_generate_mipmap;
          continue;
        }
#endif
#ifdef GL_SGIS_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample", 11))
        {
          ret = GLEW_SGIS_multisample;
          continue;
        }
#endif
#ifdef GL_SGIS_pixel_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_texture", 13))
        {
          ret = GLEW_SGIS_pixel_texture;
          continue;
        }
#endif
#ifdef GL_SGIS_point_line_texgen
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"point_line_texgen", 17))
        {
          ret = GLEW_SGIS_point_line_texgen;
          continue;
        }
#endif
#ifdef GL_SGIS_sharpen_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sharpen_texture", 15))
        {
          ret = GLEW_SGIS_sharpen_texture;
          continue;
        }
#endif
#ifdef GL_SGIS_texture4D
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture4D", 9))
        {
          ret = GLEW_SGIS_texture4D;
          continue;
        }
#endif
#ifdef GL_SGIS_texture_border_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_border_clamp", 20))
        {
          ret = GLEW_SGIS_texture_border_clamp;
          continue;
        }
#endif
#ifdef GL_SGIS_texture_edge_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_edge_clamp", 18))
        {
          ret = GLEW_SGIS_texture_edge_clamp;
          continue;
        }
#endif
#ifdef GL_SGIS_texture_filter4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_filter4", 15))
        {
          ret = GLEW_SGIS_texture_filter4;
          continue;
        }
#endif
#ifdef GL_SGIS_texture_lod
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_lod", 11))
        {
          ret = GLEW_SGIS_texture_lod;
          continue;
        }
#endif
#ifdef GL_SGIS_texture_select
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_select", 14))
        {
          ret = GLEW_SGIS_texture_select;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"SGIX_", 5))
      {
#ifdef GL_SGIX_async
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"async", 5))
        {
          ret = GLEW_SGIX_async;
          continue;
        }
#endif
#ifdef GL_SGIX_async_histogram
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"async_histogram", 15))
        {
          ret = GLEW_SGIX_async_histogram;
          continue;
        }
#endif
#ifdef GL_SGIX_async_pixel
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"async_pixel", 11))
        {
          ret = GLEW_SGIX_async_pixel;
          continue;
        }
#endif
#ifdef GL_SGIX_blend_alpha_minmax
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blend_alpha_minmax", 18))
        {
          ret = GLEW_SGIX_blend_alpha_minmax;
          continue;
        }
#endif
#ifdef GL_SGIX_clipmap
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"clipmap", 7))
        {
          ret = GLEW_SGIX_clipmap;
          continue;
        }
#endif
#ifdef GL_SGIX_convolution_accuracy
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"convolution_accuracy", 20))
        {
          ret = GLEW_SGIX_convolution_accuracy;
          continue;
        }
#endif
#ifdef GL_SGIX_depth_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_texture", 13))
        {
          ret = GLEW_SGIX_depth_texture;
          continue;
        }
#endif
#ifdef GL_SGIX_flush_raster
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"flush_raster", 12))
        {
          ret = GLEW_SGIX_flush_raster;
          continue;
        }
#endif
#ifdef GL_SGIX_fog_offset
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fog_offset", 10))
        {
          ret = GLEW_SGIX_fog_offset;
          continue;
        }
#endif
#ifdef GL_SGIX_fog_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fog_texture", 11))
        {
          ret = GLEW_SGIX_fog_texture;
          continue;
        }
#endif
#ifdef GL_SGIX_fragment_specular_lighting
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fragment_specular_lighting", 26))
        {
          ret = GLEW_SGIX_fragment_specular_lighting;
          continue;
        }
#endif
#ifdef GL_SGIX_framezoom
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framezoom", 9))
        {
          ret = GLEW_SGIX_framezoom;
          continue;
        }
#endif
#ifdef GL_SGIX_interlace
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"interlace", 9))
        {
          ret = GLEW_SGIX_interlace;
          continue;
        }
#endif
#ifdef GL_SGIX_ir_instrument1
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"ir_instrument1", 14))
        {
          ret = GLEW_SGIX_ir_instrument1;
          continue;
        }
#endif
#ifdef GL_SGIX_list_priority
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"list_priority", 13))
        {
          ret = GLEW_SGIX_list_priority;
          continue;
        }
#endif
#ifdef GL_SGIX_pixel_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_texture", 13))
        {
          ret = GLEW_SGIX_pixel_texture;
          continue;
        }
#endif
#ifdef GL_SGIX_pixel_texture_bits
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_texture_bits", 18))
        {
          ret = GLEW_SGIX_pixel_texture_bits;
          continue;
        }
#endif
#ifdef GL_SGIX_reference_plane
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"reference_plane", 15))
        {
          ret = GLEW_SGIX_reference_plane;
          continue;
        }
#endif
#ifdef GL_SGIX_resample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"resample", 8))
        {
          ret = GLEW_SGIX_resample;
          continue;
        }
#endif
#ifdef GL_SGIX_shadow
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shadow", 6))
        {
          ret = GLEW_SGIX_shadow;
          continue;
        }
#endif
#ifdef GL_SGIX_shadow_ambient
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shadow_ambient", 14))
        {
          ret = GLEW_SGIX_shadow_ambient;
          continue;
        }
#endif
#ifdef GL_SGIX_sprite
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sprite", 6))
        {
          ret = GLEW_SGIX_sprite;
          continue;
        }
#endif
#ifdef GL_SGIX_tag_sample_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"tag_sample_buffer", 17))
        {
          ret = GLEW_SGIX_tag_sample_buffer;
          continue;
        }
#endif
#ifdef GL_SGIX_texture_add_env
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_add_env", 15))
        {
          ret = GLEW_SGIX_texture_add_env;
          continue;
        }
#endif
#ifdef GL_SGIX_texture_coordinate_clamp
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_coordinate_clamp", 24))
        {
          ret = GLEW_SGIX_texture_coordinate_clamp;
          continue;
        }
#endif
#ifdef GL_SGIX_texture_lod_bias
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_lod_bias", 16))
        {
          ret = GLEW_SGIX_texture_lod_bias;
          continue;
        }
#endif
#ifdef GL_SGIX_texture_multi_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_multi_buffer", 20))
        {
          ret = GLEW_SGIX_texture_multi_buffer;
          continue;
        }
#endif
#ifdef GL_SGIX_texture_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_range", 13))
        {
          ret = GLEW_SGIX_texture_range;
          continue;
        }
#endif
#ifdef GL_SGIX_texture_scale_bias
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_scale_bias", 18))
        {
          ret = GLEW_SGIX_texture_scale_bias;
          continue;
        }
#endif
#ifdef GL_SGIX_vertex_preclip
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_preclip", 14))
        {
          ret = GLEW_SGIX_vertex_preclip;
          continue;
        }
#endif
#ifdef GL_SGIX_vertex_preclip_hint
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_preclip_hint", 19))
        {
          ret = GLEW_SGIX_vertex_preclip_hint;
          continue;
        }
#endif
#ifdef GL_SGIX_ycrcb
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"ycrcb", 5))
        {
          ret = GLEW_SGIX_ycrcb;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"SGI_", 4))
      {
#ifdef GL_SGI_color_matrix
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"color_matrix", 12))
        {
          ret = GLEW_SGI_color_matrix;
          continue;
        }
#endif
#ifdef GL_SGI_color_table
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"color_table", 11))
        {
          ret = GLEW_SGI_color_table;
          continue;
        }
#endif
#ifdef GL_SGI_texture_color_table
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_color_table", 19))
        {
          ret = GLEW_SGI_texture_color_table;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"SUNX_", 5))
      {
#ifdef GL_SUNX_constant_data
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"constant_data", 13))
        {
          ret = GLEW_SUNX_constant_data;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"SUN_", 4))
      {
#ifdef GL_SUN_convolution_border_modes
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"convolution_border_modes", 24))
        {
          ret = GLEW_SUN_convolution_border_modes;
          continue;
        }
#endif
#ifdef GL_SUN_global_alpha
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"global_alpha", 12))
        {
          ret = GLEW_SUN_global_alpha;
          continue;
        }
#endif
#ifdef GL_SUN_mesh_array
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"mesh_array", 10))
        {
          ret = GLEW_SUN_mesh_array;
          continue;
        }
#endif
#ifdef GL_SUN_read_video_pixels
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"read_video_pixels", 17))
        {
          ret = GLEW_SUN_read_video_pixels;
          continue;
        }
#endif
#ifdef GL_SUN_slice_accum
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"slice_accum", 11))
        {
          ret = GLEW_SUN_slice_accum;
          continue;
        }
#endif
#ifdef GL_SUN_triangle_list
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"triangle_list", 13))
        {
          ret = GLEW_SUN_triangle_list;
          continue;
        }
#endif
#ifdef GL_SUN_vertex
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex", 6))
        {
          ret = GLEW_SUN_vertex;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"WIN_", 4))
      {
#ifdef GL_WIN_phong_shading
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"phong_shading", 13))
        {
          ret = GLEW_WIN_phong_shading;
          continue;
        }
#endif
#ifdef GL_WIN_specular_fog
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"specular_fog", 12))
        {
          ret = GLEW_WIN_specular_fog;
          continue;
        }
#endif
#ifdef GL_WIN_swap_hint
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_hint", 9))
        {
          ret = GLEW_WIN_swap_hint;
          continue;
        }
#endif
      }
    }
    ret = (len == 0);
  }
  return ret;
}

#if defined(_WIN32)

#if defined(GLEW_MX)
GLboolean GLEWAPIENTRY wglewContextIsSupported (const WGLEWContext* ctx, const char* name)
#else
GLboolean GLEWAPIENTRY wglewIsSupported (const char* name)
#endif
{
  const GLubyte* pos = (const GLubyte*)name;
  GLuint len = _glewStrLen(pos);
  GLboolean ret = GL_TRUE;
  while (ret && len > 0)
  {
    if (_glewStrSame1(&pos, &len, (const GLubyte*)"WGL_", 4))
    {
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"3DFX_", 5))
      {
#ifdef WGL_3DFX_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample", 11))
        {
          ret = WGLEW_3DFX_multisample;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"3DL_", 4))
      {
#ifdef WGL_3DL_stereo_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"stereo_control", 14))
        {
          ret = WGLEW_3DL_stereo_control;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"AMD_", 4))
      {
#ifdef WGL_AMD_gpu_association
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_association", 15))
        {
          ret = WGLEW_AMD_gpu_association;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"ARB_", 4))
      {
#ifdef WGL_ARB_buffer_region
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"buffer_region", 13))
        {
          ret = WGLEW_ARB_buffer_region;
          continue;
        }
#endif
#ifdef WGL_ARB_context_flush_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"context_flush_control", 21))
        {
          ret = WGLEW_ARB_context_flush_control;
          continue;
        }
#endif
#ifdef WGL_ARB_create_context
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"create_context", 14))
        {
          ret = WGLEW_ARB_create_context;
          continue;
        }
#endif
#ifdef WGL_ARB_create_context_profile
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"create_context_profile", 22))
        {
          ret = WGLEW_ARB_create_context_profile;
          continue;
        }
#endif
#ifdef WGL_ARB_create_context_robustness
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"create_context_robustness", 25))
        {
          ret = WGLEW_ARB_create_context_robustness;
          continue;
        }
#endif
#ifdef WGL_ARB_extensions_string
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"extensions_string", 17))
        {
          ret = WGLEW_ARB_extensions_string;
          continue;
        }
#endif
#ifdef WGL_ARB_framebuffer_sRGB
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_sRGB", 16))
        {
          ret = WGLEW_ARB_framebuffer_sRGB;
          continue;
        }
#endif
#ifdef WGL_ARB_make_current_read
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"make_current_read", 17))
        {
          ret = WGLEW_ARB_make_current_read;
          continue;
        }
#endif
#ifdef WGL_ARB_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample", 11))
        {
          ret = WGLEW_ARB_multisample;
          continue;
        }
#endif
#ifdef WGL_ARB_pbuffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pbuffer", 7))
        {
          ret = WGLEW_ARB_pbuffer;
          continue;
        }
#endif
#ifdef WGL_ARB_pixel_format
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_format", 12))
        {
          ret = WGLEW_ARB_pixel_format;
          continue;
        }
#endif
#ifdef WGL_ARB_pixel_format_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_format_float", 18))
        {
          ret = WGLEW_ARB_pixel_format_float;
          continue;
        }
#endif
#ifdef WGL_ARB_render_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"render_texture", 14))
        {
          ret = WGLEW_ARB_render_texture;
          continue;
        }
#endif
#ifdef WGL_ARB_robustness_application_isolation
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"robustness_application_isolation", 32))
        {
          ret = WGLEW_ARB_robustness_application_isolation;
          continue;
        }
#endif
#ifdef WGL_ARB_robustness_share_group_isolation
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"robustness_share_group_isolation", 32))
        {
          ret = WGLEW_ARB_robustness_share_group_isolation;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"ATI_", 4))
      {
#ifdef WGL_ATI_pixel_format_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_format_float", 18))
        {
          ret = WGLEW_ATI_pixel_format_float;
          continue;
        }
#endif
#ifdef WGL_ATI_render_texture_rectangle
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"render_texture_rectangle", 24))
        {
          ret = WGLEW_ATI_render_texture_rectangle;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"EXT_", 4))
      {
#ifdef WGL_EXT_create_context_es2_profile
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"create_context_es2_profile", 26))
        {
          ret = WGLEW_EXT_create_context_es2_profile;
          continue;
        }
#endif
#ifdef WGL_EXT_create_context_es_profile
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"create_context_es_profile", 25))
        {
          ret = WGLEW_EXT_create_context_es_profile;
          continue;
        }
#endif
#ifdef WGL_EXT_depth_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"depth_float", 11))
        {
          ret = WGLEW_EXT_depth_float;
          continue;
        }
#endif
#ifdef WGL_EXT_display_color_table
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"display_color_table", 19))
        {
          ret = WGLEW_EXT_display_color_table;
          continue;
        }
#endif
#ifdef WGL_EXT_extensions_string
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"extensions_string", 17))
        {
          ret = WGLEW_EXT_extensions_string;
          continue;
        }
#endif
#ifdef WGL_EXT_framebuffer_sRGB
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_sRGB", 16))
        {
          ret = WGLEW_EXT_framebuffer_sRGB;
          continue;
        }
#endif
#ifdef WGL_EXT_make_current_read
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"make_current_read", 17))
        {
          ret = WGLEW_EXT_make_current_read;
          continue;
        }
#endif
#ifdef WGL_EXT_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample", 11))
        {
          ret = WGLEW_EXT_multisample;
          continue;
        }
#endif
#ifdef WGL_EXT_pbuffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pbuffer", 7))
        {
          ret = WGLEW_EXT_pbuffer;
          continue;
        }
#endif
#ifdef WGL_EXT_pixel_format
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_format", 12))
        {
          ret = WGLEW_EXT_pixel_format;
          continue;
        }
#endif
#ifdef WGL_EXT_pixel_format_packed_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_format_packed_float", 25))
        {
          ret = WGLEW_EXT_pixel_format_packed_float;
          continue;
        }
#endif
#ifdef WGL_EXT_swap_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_control", 12))
        {
          ret = WGLEW_EXT_swap_control;
          continue;
        }
#endif
#ifdef WGL_EXT_swap_control_tear
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_control_tear", 17))
        {
          ret = WGLEW_EXT_swap_control_tear;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"I3D_", 4))
      {
#ifdef WGL_I3D_digital_video_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"digital_video_control", 21))
        {
          ret = WGLEW_I3D_digital_video_control;
          continue;
        }
#endif
#ifdef WGL_I3D_gamma
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gamma", 5))
        {
          ret = WGLEW_I3D_gamma;
          continue;
        }
#endif
#ifdef WGL_I3D_genlock
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"genlock", 7))
        {
          ret = WGLEW_I3D_genlock;
          continue;
        }
#endif
#ifdef WGL_I3D_image_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"image_buffer", 12))
        {
          ret = WGLEW_I3D_image_buffer;
          continue;
        }
#endif
#ifdef WGL_I3D_swap_frame_lock
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_frame_lock", 15))
        {
          ret = WGLEW_I3D_swap_frame_lock;
          continue;
        }
#endif
#ifdef WGL_I3D_swap_frame_usage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_frame_usage", 16))
        {
          ret = WGLEW_I3D_swap_frame_usage;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"NV_", 3))
      {
#ifdef WGL_NV_DX_interop
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"DX_interop", 10))
        {
          ret = WGLEW_NV_DX_interop;
          continue;
        }
#endif
#ifdef WGL_NV_DX_interop2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"DX_interop2", 11))
        {
          ret = WGLEW_NV_DX_interop2;
          continue;
        }
#endif
#ifdef WGL_NV_copy_image
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"copy_image", 10))
        {
          ret = WGLEW_NV_copy_image;
          continue;
        }
#endif
#ifdef WGL_NV_delay_before_swap
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"delay_before_swap", 17))
        {
          ret = WGLEW_NV_delay_before_swap;
          continue;
        }
#endif
#ifdef WGL_NV_float_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"float_buffer", 12))
        {
          ret = WGLEW_NV_float_buffer;
          continue;
        }
#endif
#ifdef WGL_NV_gpu_affinity
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_affinity", 12))
        {
          ret = WGLEW_NV_gpu_affinity;
          continue;
        }
#endif
#ifdef WGL_NV_multisample_coverage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample_coverage", 20))
        {
          ret = WGLEW_NV_multisample_coverage;
          continue;
        }
#endif
#ifdef WGL_NV_present_video
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"present_video", 13))
        {
          ret = WGLEW_NV_present_video;
          continue;
        }
#endif
#ifdef WGL_NV_render_depth_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"render_depth_texture", 20))
        {
          ret = WGLEW_NV_render_depth_texture;
          continue;
        }
#endif
#ifdef WGL_NV_render_texture_rectangle
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"render_texture_rectangle", 24))
        {
          ret = WGLEW_NV_render_texture_rectangle;
          continue;
        }
#endif
#ifdef WGL_NV_swap_group
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_group", 10))
        {
          ret = WGLEW_NV_swap_group;
          continue;
        }
#endif
#ifdef WGL_NV_vertex_array_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_range", 18))
        {
          ret = WGLEW_NV_vertex_array_range;
          continue;
        }
#endif
#ifdef WGL_NV_video_capture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"video_capture", 13))
        {
          ret = WGLEW_NV_video_capture;
          continue;
        }
#endif
#ifdef WGL_NV_video_output
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"video_output", 12))
        {
          ret = WGLEW_NV_video_output;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"OML_", 4))
      {
#ifdef WGL_OML_sync_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sync_control", 12))
        {
          ret = WGLEW_OML_sync_control;
          continue;
        }
#endif
      }
    }
    ret = (len == 0);
  }
  return ret;
}

#elif !defined(__ANDROID__) && !defined(__native_client__) && !defined(__HAIKU__) && !defined(__APPLE__) || defined(GLEW_APPLE_GLX)

#if defined(GLEW_MX)
GLboolean glxewContextIsSupported (const GLXEWContext* ctx, const char* name)
#else
GLboolean glxewIsSupported (const char* name)
#endif
{
  const GLubyte* pos = (const GLubyte*)name;
  GLuint len = _glewStrLen(pos);
  GLboolean ret = GL_TRUE;
  while (ret && len > 0)
  {
    if(_glewStrSame1(&pos, &len, (const GLubyte*)"GLX_", 4))
    {
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"VERSION_", 8))
      {
#ifdef GLX_VERSION_1_2
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"1_2", 3))
        {
          ret = GLXEW_VERSION_1_2;
          continue;
        }
#endif
#ifdef GLX_VERSION_1_3
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"1_3", 3))
        {
          ret = GLXEW_VERSION_1_3;
          continue;
        }
#endif
#ifdef GLX_VERSION_1_4
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"1_4", 3))
        {
          ret = GLXEW_VERSION_1_4;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"3DFX_", 5))
      {
#ifdef GLX_3DFX_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample", 11))
        {
          ret = GLXEW_3DFX_multisample;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"AMD_", 4))
      {
#ifdef GLX_AMD_gpu_association
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"gpu_association", 15))
        {
          ret = GLXEW_AMD_gpu_association;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"ARB_", 4))
      {
#ifdef GLX_ARB_context_flush_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"context_flush_control", 21))
        {
          ret = GLXEW_ARB_context_flush_control;
          continue;
        }
#endif
#ifdef GLX_ARB_create_context
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"create_context", 14))
        {
          ret = GLXEW_ARB_create_context;
          continue;
        }
#endif
#ifdef GLX_ARB_create_context_profile
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"create_context_profile", 22))
        {
          ret = GLXEW_ARB_create_context_profile;
          continue;
        }
#endif
#ifdef GLX_ARB_create_context_robustness
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"create_context_robustness", 25))
        {
          ret = GLXEW_ARB_create_context_robustness;
          continue;
        }
#endif
#ifdef GLX_ARB_fbconfig_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fbconfig_float", 14))
        {
          ret = GLXEW_ARB_fbconfig_float;
          continue;
        }
#endif
#ifdef GLX_ARB_framebuffer_sRGB
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_sRGB", 16))
        {
          ret = GLXEW_ARB_framebuffer_sRGB;
          continue;
        }
#endif
#ifdef GLX_ARB_get_proc_address
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"get_proc_address", 16))
        {
          ret = GLXEW_ARB_get_proc_address;
          continue;
        }
#endif
#ifdef GLX_ARB_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample", 11))
        {
          ret = GLXEW_ARB_multisample;
          continue;
        }
#endif
#ifdef GLX_ARB_robustness_application_isolation
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"robustness_application_isolation", 32))
        {
          ret = GLXEW_ARB_robustness_application_isolation;
          continue;
        }
#endif
#ifdef GLX_ARB_robustness_share_group_isolation
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"robustness_share_group_isolation", 32))
        {
          ret = GLXEW_ARB_robustness_share_group_isolation;
          continue;
        }
#endif
#ifdef GLX_ARB_vertex_buffer_object
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_buffer_object", 20))
        {
          ret = GLXEW_ARB_vertex_buffer_object;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"ATI_", 4))
      {
#ifdef GLX_ATI_pixel_format_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixel_format_float", 18))
        {
          ret = GLXEW_ATI_pixel_format_float;
          continue;
        }
#endif
#ifdef GLX_ATI_render_texture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"render_texture", 14))
        {
          ret = GLXEW_ATI_render_texture;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"EXT_", 4))
      {
#ifdef GLX_EXT_buffer_age
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"buffer_age", 10))
        {
          ret = GLXEW_EXT_buffer_age;
          continue;
        }
#endif
#ifdef GLX_EXT_create_context_es2_profile
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"create_context_es2_profile", 26))
        {
          ret = GLXEW_EXT_create_context_es2_profile;
          continue;
        }
#endif
#ifdef GLX_EXT_create_context_es_profile
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"create_context_es_profile", 25))
        {
          ret = GLXEW_EXT_create_context_es_profile;
          continue;
        }
#endif
#ifdef GLX_EXT_fbconfig_packed_float
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fbconfig_packed_float", 21))
        {
          ret = GLXEW_EXT_fbconfig_packed_float;
          continue;
        }
#endif
#ifdef GLX_EXT_framebuffer_sRGB
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"framebuffer_sRGB", 16))
        {
          ret = GLXEW_EXT_framebuffer_sRGB;
          continue;
        }
#endif
#ifdef GLX_EXT_import_context
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"import_context", 14))
        {
          ret = GLXEW_EXT_import_context;
          continue;
        }
#endif
#ifdef GLX_EXT_scene_marker
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"scene_marker", 12))
        {
          ret = GLXEW_EXT_scene_marker;
          continue;
        }
#endif
#ifdef GLX_EXT_stereo_tree
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"stereo_tree", 11))
        {
          ret = GLXEW_EXT_stereo_tree;
          continue;
        }
#endif
#ifdef GLX_EXT_swap_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_control", 12))
        {
          ret = GLXEW_EXT_swap_control;
          continue;
        }
#endif
#ifdef GLX_EXT_swap_control_tear
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_control_tear", 17))
        {
          ret = GLXEW_EXT_swap_control_tear;
          continue;
        }
#endif
#ifdef GLX_EXT_texture_from_pixmap
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"texture_from_pixmap", 19))
        {
          ret = GLXEW_EXT_texture_from_pixmap;
          continue;
        }
#endif
#ifdef GLX_EXT_visual_info
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"visual_info", 11))
        {
          ret = GLXEW_EXT_visual_info;
          continue;
        }
#endif
#ifdef GLX_EXT_visual_rating
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"visual_rating", 13))
        {
          ret = GLXEW_EXT_visual_rating;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"INTEL_", 6))
      {
#ifdef GLX_INTEL_swap_event
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_event", 10))
        {
          ret = GLXEW_INTEL_swap_event;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"MESA_", 5))
      {
#ifdef GLX_MESA_agp_offset
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"agp_offset", 10))
        {
          ret = GLXEW_MESA_agp_offset;
          continue;
        }
#endif
#ifdef GLX_MESA_copy_sub_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"copy_sub_buffer", 15))
        {
          ret = GLXEW_MESA_copy_sub_buffer;
          continue;
        }
#endif
#ifdef GLX_MESA_pixmap_colormap
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pixmap_colormap", 15))
        {
          ret = GLXEW_MESA_pixmap_colormap;
          continue;
        }
#endif
#ifdef GLX_MESA_query_renderer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"query_renderer", 14))
        {
          ret = GLXEW_MESA_query_renderer;
          continue;
        }
#endif
#ifdef GLX_MESA_release_buffers
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"release_buffers", 15))
        {
          ret = GLXEW_MESA_release_buffers;
          continue;
        }
#endif
#ifdef GLX_MESA_set_3dfx_mode
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"set_3dfx_mode", 13))
        {
          ret = GLXEW_MESA_set_3dfx_mode;
          continue;
        }
#endif
#ifdef GLX_MESA_swap_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_control", 12))
        {
          ret = GLXEW_MESA_swap_control;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"NV_", 3))
      {
#ifdef GLX_NV_copy_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"copy_buffer", 11))
        {
          ret = GLXEW_NV_copy_buffer;
          continue;
        }
#endif
#ifdef GLX_NV_copy_image
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"copy_image", 10))
        {
          ret = GLXEW_NV_copy_image;
          continue;
        }
#endif
#ifdef GLX_NV_delay_before_swap
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"delay_before_swap", 17))
        {
          ret = GLXEW_NV_delay_before_swap;
          continue;
        }
#endif
#ifdef GLX_NV_float_buffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"float_buffer", 12))
        {
          ret = GLXEW_NV_float_buffer;
          continue;
        }
#endif
#ifdef GLX_NV_multisample_coverage
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample_coverage", 20))
        {
          ret = GLXEW_NV_multisample_coverage;
          continue;
        }
#endif
#ifdef GLX_NV_present_video
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"present_video", 13))
        {
          ret = GLXEW_NV_present_video;
          continue;
        }
#endif
#ifdef GLX_NV_swap_group
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_group", 10))
        {
          ret = GLXEW_NV_swap_group;
          continue;
        }
#endif
#ifdef GLX_NV_vertex_array_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"vertex_array_range", 18))
        {
          ret = GLXEW_NV_vertex_array_range;
          continue;
        }
#endif
#ifdef GLX_NV_video_capture
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"video_capture", 13))
        {
          ret = GLXEW_NV_video_capture;
          continue;
        }
#endif
#ifdef GLX_NV_video_out
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"video_out", 9))
        {
          ret = GLXEW_NV_video_out;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"OML_", 4))
      {
#ifdef GLX_OML_swap_method
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_method", 11))
        {
          ret = GLXEW_OML_swap_method;
          continue;
        }
#endif
#ifdef GLX_OML_sync_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"sync_control", 12))
        {
          ret = GLXEW_OML_sync_control;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"SGIS_", 5))
      {
#ifdef GLX_SGIS_blended_overlay
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"blended_overlay", 15))
        {
          ret = GLXEW_SGIS_blended_overlay;
          continue;
        }
#endif
#ifdef GLX_SGIS_color_range
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"color_range", 11))
        {
          ret = GLXEW_SGIS_color_range;
          continue;
        }
#endif
#ifdef GLX_SGIS_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"multisample", 11))
        {
          ret = GLXEW_SGIS_multisample;
          continue;
        }
#endif
#ifdef GLX_SGIS_shared_multisample
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"shared_multisample", 18))
        {
          ret = GLXEW_SGIS_shared_multisample;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"SGIX_", 5))
      {
#ifdef GLX_SGIX_fbconfig
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"fbconfig", 8))
        {
          ret = GLXEW_SGIX_fbconfig;
          continue;
        }
#endif
#ifdef GLX_SGIX_hyperpipe
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"hyperpipe", 9))
        {
          ret = GLXEW_SGIX_hyperpipe;
          continue;
        }
#endif
#ifdef GLX_SGIX_pbuffer
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"pbuffer", 7))
        {
          ret = GLXEW_SGIX_pbuffer;
          continue;
        }
#endif
#ifdef GLX_SGIX_swap_barrier
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_barrier", 12))
        {
          ret = GLXEW_SGIX_swap_barrier;
          continue;
        }
#endif
#ifdef GLX_SGIX_swap_group
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_group", 10))
        {
          ret = GLXEW_SGIX_swap_group;
          continue;
        }
#endif
#ifdef GLX_SGIX_video_resize
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"video_resize", 12))
        {
          ret = GLXEW_SGIX_video_resize;
          continue;
        }
#endif
#ifdef GLX_SGIX_visual_select_group
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"visual_select_group", 19))
        {
          ret = GLXEW_SGIX_visual_select_group;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"SGI_", 4))
      {
#ifdef GLX_SGI_cushion
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"cushion", 7))
        {
          ret = GLXEW_SGI_cushion;
          continue;
        }
#endif
#ifdef GLX_SGI_make_current_read
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"make_current_read", 17))
        {
          ret = GLXEW_SGI_make_current_read;
          continue;
        }
#endif
#ifdef GLX_SGI_swap_control
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"swap_control", 12))
        {
          ret = GLXEW_SGI_swap_control;
          continue;
        }
#endif
#ifdef GLX_SGI_video_sync
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"video_sync", 10))
        {
          ret = GLXEW_SGI_video_sync;
          continue;
        }
#endif
      }
      if (_glewStrSame2(&pos, &len, (const GLubyte*)"SUN_", 4))
      {
#ifdef GLX_SUN_get_transparent_index
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"get_transparent_index", 21))
        {
          ret = GLXEW_SUN_get_transparent_index;
          continue;
        }
#endif
#ifdef GLX_SUN_video_resize
        if (_glewStrSame3(&pos, &len, (const GLubyte*)"video_resize", 12))
        {
          ret = GLXEW_SUN_video_resize;
          continue;
        }
#endif
      }
    }
    ret = (len == 0);
  }
  return ret;
}

#endif /* _WIN32 */
