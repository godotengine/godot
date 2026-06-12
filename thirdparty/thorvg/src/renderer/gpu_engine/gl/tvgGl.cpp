/*
 * Copyright (c) 2025 - 2026 ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "tvgGl.h"
#include "tvgCommon.h"

#ifdef __EMSCRIPTEN__

bool glInit()
{
    return true;
}


bool glTerm()
{
    return true;
}

#else //__EMSCRIPTEN__

#if defined(_WIN32) && !defined(__CYGWIN__)

#ifdef THORVG_GL_TARGET_GL
    typedef PROC(WINAPI *PFNGLGETPROCADDRESSPROC)(LPCSTR);
    static PFNGLGETPROCADDRESSPROC glGetProcAddress = NULL;
#endif

static HMODULE _libGL = NULL;
#ifdef THORVG_GL_TARGET_GLES
    static HMODULE _libEGL = NULL;
#endif

static bool _glLoad()
{
#ifdef THORVG_GL_TARGET_GL
    _libGL = LoadLibraryW(L"opengl32.dll");
    if (!_libGL) {
        TVGERR("GL_ENGINE", "Cannot find the gl library.");
        return false;
    }
    if (!glGetProcAddress) glGetProcAddress = (PFNGLGETPROCADDRESSPROC)GetProcAddress(_libGL, "wglGetProcAddress");
    if (!glGetProcAddress) glGetProcAddress = (PFNGLGETPROCADDRESSPROC)GetProcAddress(_libGL, "wglGetProcAddressARB");
    if (!glGetProcAddress) return false;
#else
    if (!_libGL) _libGL = LoadLibraryW(L"GLESv2.dll");
    if (!_libGL) _libGL = LoadLibraryW(L"libGLESv2.dll");
    if (!_libGL) {
        TVGERR("GL_ENGINE", "Cannot find gles library.");
        return false;
    }
#endif
    return true;
}

// load opengl proc address from dll or from wglGetProcAddress
static PROC _getProcAddress(const char* procName)
{
    auto procHandle = GetProcAddress(_libGL, procName);
#ifdef THORVG_GL_TARGET_GL
    if (!procHandle) procHandle = glGetProcAddress(procName);
#endif
    return procHandle;
}

#elif defined(__linux__)

#include <dlfcn.h>

#ifdef THORVG_GL_TARGET_GL
    typedef void* (*PFNGLGETPROCADDRESSPROC)(const char*);
    static PFNGLGETPROCADDRESSPROC glGetProcAddress = nullptr;
#endif

static void* _libGL = nullptr;
#ifdef THORVG_GL_TARGET_GLES
    static void* _libEGL = nullptr;
#endif

static bool _glLoad()
{
#ifdef THORVG_GL_TARGET_GL
    _libGL = dlopen("libGL.so", RTLD_LAZY);
    if (!_libGL) _libGL = dlopen("libGL.so.4", RTLD_LAZY);
    if (!_libGL) _libGL = dlopen("libGL.so.3", RTLD_LAZY);
    if (!_libGL) {
        TVGERR("GL_ENGINE", "Cannot find the gl library.");
        return false;
    }
    if (!glGetProcAddress) glGetProcAddress = (PFNGLGETPROCADDRESSPROC)dlsym(_libGL, "glXGetProcAddress");
    if (!glGetProcAddress) glGetProcAddress = (PFNGLGETPROCADDRESSPROC)dlsym(_libGL, "glXGetProcAddressARB");
    if (!glGetProcAddress) return false;
#else
    if (!_libGL) _libGL = dlopen("libGLESv2.so", RTLD_LAZY);
    if (!_libGL) _libGL = dlopen("libGLESv2.so.2.0", RTLD_LAZY);  // sometimes used in Mesa
    if (!_libGL) _libGL = dlopen("libGLESv2.so.2", RTLD_LAZY);
    if (!_libGL) {
        TVGERR("GL_ENGINE", "Cannot find the gles library.");
        return false;
    }
#endif
    return true;
}


static void* _getProcAddress(const char* procName)
{
#ifdef THORVG_GL_TARGET_GL
    return glGetProcAddress(procName);
#else
    return dlsym(_libGL, procName);
#endif
}

#elif defined(__APPLE__) || defined(__MACH__)

#include <dlfcn.h>

static void* _libGL = nullptr;
#ifdef THORVG_GL_TARGET_GLES
    static void* _libEGL = nullptr;
#endif

static bool _glLoad()
{
    if (!_libGL) _libGL = dlopen("/Library/Frameworks/OpenGL.framework/OpenGL", RTLD_LAZY);
    if (!_libGL) _libGL = dlopen("/System/Library/Frameworks/OpenGL.framework/OpenGL", RTLD_LAZY);
    if (_libGL) return true;
    TVGERR("GL_ENGINE", "Cannot find gl library.");
    return false;
}


static void* _getProcAddress(const char* procName)
{
    return dlsym(_libGL, procName);
}

#endif

#define GL_FUNCTION_FETCH(procName, procType)                   \
    procName = (procType)_getProcAddress(#procName);            \
    if (!procName) {                                            \
        TVGERR("GL_ENGINE", "%s is not supported.", #procName); \
        return false;                                           \
    }

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

//GL_VERSION_1_0
PFNGLCULLFACEPROC               glCullFace;
PFNGLFRONTFACEPROC              glFrontFace;
PFNGLSCISSORPROC                glScissor;
PFNGLTEXPARAMETERIPROC          glTexParameteri;
PFNGLTEXIMAGE2DPROC             glTexImage2D;
PFNGLDRAWBUFFERPROC             glDrawBuffer;
PFNGLCLEARPROC                  glClear;
PFNGLCLEARCOLORPROC             glClearColor;
PFNGLCLEARSTENCILPROC           glClearStencil;
PFNGLCLEARDEPTHPROC             glClearDepth;
PFNGLCLEARDEPTHFPROC            glClearDepthf; // GLES
PFNGLCOLORMASKPROC              glColorMask;
PFNGLDEPTHMASKPROC              glDepthMask;
PFNGLDISABLEPROC                glDisable;
PFNGLENABLEPROC                 glEnable;
PFNGLBLENDFUNCPROC              glBlendFunc;
PFNGLSTENCILFUNCPROC            glStencilFunc;
PFNGLSTENCILOPPROC              glStencilOp;
PFNGLDEPTHFUNCPROC              glDepthFunc;
PFNGLGETERRORPROC               glGetError;
PFNGLGETINTEGERVPROC            glGetIntegerv;
PFNGLGETSTRINGPROC              glGetString;
PFNGLVIEWPORTPROC               glViewport;
//PFNGLHINTPROC                   glHint;
//PFNGLLINEWIDTHPROC              glLineWidth;
//PFNGLPOINTSIZEPROC              glPointSize;
//PFNGLPOLYGONMODEPROC            glPolygonMode;
//PFNGLTEXPARAMETERFPROC          glTexParameterf;
//PFNGLTEXPARAMETERFVPROC         glTexParameterfv;
//PFNGLTEXPARAMETERIVPROC         glTexParameteriv;
//PFNGLTEXIMAGE1DPROC             glTexImage1D;
//PFNGLSTENCILMASKPROC            glStencilMask;
//PFNGLFINISHPROC                 glFinish;
//PFNGLFLUSHPROC                  glFlush;
//PFNGLLOGICOPPROC                glLogicOp = nullptr
//PFNGLPIXELSTOREFPROC            glPixelStoref;
//PFNGLPIXELSTOREIPROC            glPixelStorei;
//PFNGLREADBUFFERPROC             glReadBuffer;
//PFNGLREADPIXELSPROC             glReadPixels;
//PFNGLGETBOOLEANVPROC            glGetBooleanv;
//PFNGLGETDOUBLEVPROC             glGetDoublev;
//PFNGLGETFLOATVPROC              glGetFloatv;
//PFNGLGETTEXIMAGEPROC            glGetTexImage;
//PFNGLGETTEXPARAMETERFVPROC      glGetTexParameterfv;
//PFNGLGETTEXPARAMETERIVPROC      glGetTexParameteriv;
//PFNGLGETTEXLEVELPARAMETERFVPROC glGetTexLevelParameterfv;
//PFNGLGETTEXLEVELPARAMETERIVPROC glGetTexLevelParameteriv;
//PFNGLISENABLEDPROC              glIsEnabled;
//PFNGLDEPTHRANGEPROC             glDepthRange;

//GL_VERSION_1_1
PFNGLDRAWELEMENTSPROC      glDrawElements;
PFNGLBINDTEXTUREPROC       glBindTexture;
PFNGLDELETETEXTURESPROC    glDeleteTextures;
PFNGLGENTEXTURESPROC       glGenTextures;
//PFNGLDRAWARRAYSPROC        glDrawArrays;
//PFNGLGETPOINTERVPROC       glGetPointerv;
//PFNGLPOLYGONOFFSETPROC     glPolygonOffset;
//PFNGLCOPYTEXIMAGE1DPROC    glCopyTexImage1D;
//PFNGLCOPYTEXIMAGE2DPROC    glCopyTexImage2D;
//PFNGLCOPYTEXSUBIMAGE1DPROC glCopyTexSubImage1D;
//PFNGLCOPYTEXSUBIMAGE2DPROC glCopyTexSubImage2D;
//PFNGLTEXSUBIMAGE1DPROC     glTexSubImage1D;
//PFNGLTEXSUBIMAGE2DPROC     glTexSubImage2D;
//PFNGLISTEXTUREPROC         glIsTexture;

//GL_VERSION_1_2
//PFNGLDRAWRANGEELEMENTSPROC glDrawRangeElements;
//PFNGLTEXIMAGE3DPROC        glTexImage3D;
//PFNGLTEXSUBIMAGE3DPROC     glTexSubImage3D;
//PFNGLCOPYTEXSUBIMAGE3DPROC glCopyTexSubImage3D;

//GL_VERSION_1_3
PFNGLACTIVETEXTUREPROC           glActiveTexture;
//PFNGLSAMPLECOVERAGEPROC          glSampleCoverage;
//PFNGLCOMPRESSEDTEXIMAGE3DPROC    glCompressedTexImage3D;
//PFNGLCOMPRESSEDTEXIMAGE2DPROC    glCompressedTexImage2D;
//PFNGLCOMPRESSEDTEXIMAGE1DPROC    glCompressedTexImage1D;
//PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC glCompressedTexSubImage3D;
//PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC glCompressedTexSubImage2D;
//PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC glCompressedTexSubImage1D;
//PFNGLGETCOMPRESSEDTEXIMAGEPROC   glGetCompressedTexImage;

//GL_VERSION_1_4
PFNGLBLENDEQUATIONPROC     glBlendEquation;
//PFNGLBLENDFUNCSEPARATEPROC glBlendFuncSeparate;
//PFNGLMULTIDRAWARRAYSPROC   glMultiDrawArrays;
//PFNGLMULTIDRAWELEMENTSPROC glMultiDrawElements;
//PFNGLPOINTPARAMETERFPROC   glPointParameterf;
//PFNGLPOINTPARAMETERFVPROC  glPointParameterfv;
//PFNGLPOINTPARAMETERIPROC   glPointParameteri;
//PFNGLPOINTPARAMETERIVPROC  glPointParameteriv;
//PFNGLBLENDCOLORPROC        glBlendColor;

//GL_VERSION_1_5
PFNGLBINDBUFFERPROC           glBindBuffer;
PFNGLDELETEBUFFERSPROC        glDeleteBuffers;
PFNGLGENBUFFERSPROC           glGenBuffers;
PFNGLBUFFERDATAPROC           glBufferData;
//PFNGLGENQUERIESPROC           glGenQueries;
//PFNGLDELETEQUERIESPROC        glDeleteQueries;
//PFNGLISQUERYPROC              glIsQuery;
//PFNGLBEGINQUERYPROC           glBeginQuery;
//PFNGLENDQUERYPROC             glEndQuery;
//PFNGLGETQUERYIVPROC           glGetQueryiv;
//PFNGLGETQUERYOBJECTIVPROC     glGetQueryObjectiv;
//PFNGLGETQUERYOBJECTUIVPROC    glGetQueryObjectuiv;
//PFNGLISBUFFERPROC             glIsBuffer;
//PFNGLBUFFERSUBDATAPROC        glBufferSubData;
//PFNGLGETBUFFERSUBDATAPROC     glGetBufferSubData;
//PFNGLMAPBUFFERPROC            glMapBuffer;
//PFNGLUNMAPBUFFERPROC          glUnmapBuffer;
//PFNGLGETBUFFERPARAMETERIVPROC glGetBufferParameteriv;
//PFNGLGETBUFFERPOINTERVPROC    glGetBufferPointerv;

//GL_VERSION_2_0
PFNGLDRAWBUFFERSPROC              glDrawBuffers;
PFNGLSTENCILOPSEPARATEPROC        glStencilOpSeparate;
PFNGLSTENCILFUNCSEPARATEPROC      glStencilFuncSeparate;
PFNGLATTACHSHADERPROC             glAttachShader;
PFNGLCOMPILESHADERPROC            glCompileShader;
PFNGLCREATEPROGRAMPROC            glCreateProgram;
PFNGLCREATESHADERPROC             glCreateShader;
PFNGLDELETEPROGRAMPROC            glDeleteProgram;
PFNGLDELETESHADERPROC             glDeleteShader;
PFNGLDISABLEVERTEXATTRIBARRAYPROC glDisableVertexAttribArray;
PFNGLENABLEVERTEXATTRIBARRAYPROC  glEnableVertexAttribArray;
PFNGLGETATTRIBLOCATIONPROC        glGetAttribLocation;
PFNGLGETPROGRAMIVPROC             glGetProgramiv;
PFNGLGETPROGRAMINFOLOGPROC        glGetProgramInfoLog;
PFNGLGETSHADERIVPROC              glGetShaderiv;
PFNGLGETSHADERINFOLOGPROC         glGetShaderInfoLog;
PFNGLGETUNIFORMLOCATIONPROC       glGetUniformLocation;
PFNGLLINKPROGRAMPROC              glLinkProgram;
PFNGLSHADERSOURCEPROC             glShaderSource;
PFNGLUSEPROGRAMPROC               glUseProgram;
PFNGLUNIFORM1FPROC                glUniform1f;
PFNGLUNIFORM1FVPROC               glUniform1fv;
PFNGLUNIFORM2FVPROC               glUniform2fv;
PFNGLUNIFORM3FVPROC               glUniform3fv;
PFNGLUNIFORM4FVPROC               glUniform4fv;
PFNGLUNIFORM1IVPROC               glUniform1iv;
PFNGLUNIFORM2IVPROC               glUniform2iv;
PFNGLUNIFORM3IVPROC               glUniform3iv;
PFNGLUNIFORM4IVPROC               glUniform4iv;
//PFNGLUNIFORMMATRIX4FVPROC         glUniformMatrix4fv;
PFNGLVERTEXATTRIBPOINTERPROC      glVertexAttribPointer;
//PFNGLBLENDEQUATIONSEPARATEPROC    glBlendEquationSeparate;
//PFNGLSTENCILMASKSEPARATEPROC      glStencilMaskSeparate;
//PFNGLBINDATTRIBLOCATIONPROC       glBindAttribLocation;
//PFNGLDETACHSHADERPROC             glDetachShader;
//PFNGLGETACTIVEATTRIBPROC          glGetActiveAttrib;
//PFNGLGETACTIVEUNIFORMPROC         glGetActiveUniform;
//PFNGLGETATTACHEDSHADERSPROC       glGetAttachedShaders;
//PFNGLGETSHADERSOURCEPROC          glGetShaderSource;
//PFNGLGETUNIFORMFVPROC             glGetUniformfv;
//PFNGLGETUNIFORMIVPROC             glGetUniformiv;
//PFNGLGETVERTEXATTRIBDVPROC        glGetVertexAttribdv;
//PFNGLGETVERTEXATTRIBFVPROC        glGetVertexAttribfv;
//PFNGLGETVERTEXATTRIBIVPROC        glGetVertexAttribiv;
//PFNGLGETVERTEXATTRIBPOINTERVPROC  glGetVertexAttribPointerv;
//PFNGLISPROGRAMPROC                glIsProgram;
//PFNGLISSHADERPROC                 glIsShader;
//PFNGLUNIFORM2FPROC                glUniform2f;
//PFNGLUNIFORM3FPROC                glUniform3f;
//PFNGLUNIFORM4FPROC                glUniform4f;
//PFNGLUNIFORM1IPROC                glUniform1i;
//PFNGLUNIFORM2IPROC                glUniform2i;
//PFNGLUNIFORM3IPROC                glUniform3i;
//PFNGLUNIFORM4IPROC                glUniform4i;
//PFNGLUNIFORMMATRIX2FVPROC         glUniformMatrix2fv;
PFNGLUNIFORMMATRIX3FVPROC          glUniformMatrix3fv;
//PFNGLVALIDATEPROGRAMPROC          glValidateProgram;
//PFNGLVERTEXATTRIB1DPROC           glVertexAttrib1d;
//PFNGLVERTEXATTRIB1DVPROC          glVertexAttrib1dv;
//PFNGLVERTEXATTRIB1FPROC           glVertexAttrib1f;
//PFNGLVERTEXATTRIB1FVPROC          glVertexAttrib1fv;
//PFNGLVERTEXATTRIB1SPROC           glVertexAttrib1s;
//PFNGLVERTEXATTRIB1SVPROC          glVertexAttrib1sv;
//PFNGLVERTEXATTRIB2DPROC           glVertexAttrib2d;
//PFNGLVERTEXATTRIB2DVPROC          glVertexAttrib2dv;
//PFNGLVERTEXATTRIB2FPROC           glVertexAttrib2f;
//PFNGLVERTEXATTRIB2FVPROC          glVertexAttrib2fv;
//PFNGLVERTEXATTRIB2SPROC           glVertexAttrib2s;
//PFNGLVERTEXATTRIB2SVPROC          glVertexAttrib2sv;
//PFNGLVERTEXATTRIB3DPROC           glVertexAttrib3d;
//PFNGLVERTEXATTRIB3DVPROC          glVertexAttrib3dv;
//PFNGLVERTEXATTRIB3FPROC           glVertexAttrib3f;
//PFNGLVERTEXATTRIB3FVPROC          glVertexAttrib3fv;
//PFNGLVERTEXATTRIB3SPROC           glVertexAttrib3s;
//PFNGLVERTEXATTRIB3SVPROC          glVertexAttrib3sv;
//PFNGLVERTEXATTRIB4NBVPROC         glVertexAttrib4Nbv;
//PFNGLVERTEXATTRIB4NIVPROC         glVertexAttrib4Niv;
//PFNGLVERTEXATTRIB4NSVPROC         glVertexAttrib4Nsv;
//PFNGLVERTEXATTRIB4NUBPROC         glVertexAttrib4Nub;
//PFNGLVERTEXATTRIB4NUBVPROC        glVertexAttrib4Nubv;
//PFNGLVERTEXATTRIB4NUIVPROC        glVertexAttrib4Nuiv;
//PFNGLVERTEXATTRIB4NUSVPROC        glVertexAttrib4Nusv;
//PFNGLVERTEXATTRIB4BVPROC          glVertexAttrib4bv;
//PFNGLVERTEXATTRIB4DPROC           glVertexAttrib4d;
//PFNGLVERTEXATTRIB4DVPROC          glVertexAttrib4dv;
PFNGLVERTEXATTRIB4FPROC             glVertexAttrib4f;
//PFNGLVERTEXATTRIB4FVPROC          glVertexAttrib4fv;
//PFNGLVERTEXATTRIB4IVPROC          glVertexAttrib4iv;
//PFNGLVERTEXATTRIB4SPROC           glVertexAttrib4s;
//PFNGLVERTEXATTRIB4SVPROC          glVertexAttrib4sv;
//PFNGLVERTEXATTRIB4UBVPROC         glVertexAttrib4ubv;
//PFNGLVERTEXATTRIB4UIVPROC         glVertexAttrib4uiv;
//PFNGLVERTEXATTRIB4USVPROC         glVertexAttrib4usv;

//GL_VERSION_2_1
//PFNGLUNIFORMMATRIX2X3FVPROC glUniformMatrix2x3fv;
//PFNGLUNIFORMMATRIX3X2FVPROC glUniformMatrix3x2fv;
//PFNGLUNIFORMMATRIX2X4FVPROC glUniformMatrix2x4fv;
//PFNGLUNIFORMMATRIX4X2FVPROC glUniformMatrix4x2fv;
//PFNGLUNIFORMMATRIX3X4FVPROC glUniformMatrix3x4fv;
//PFNGLUNIFORMMATRIX4X3FVPROC glUniformMatrix4x3fv;

//GL_VERSION_3_0
PFNGLBINDBUFFERRANGEPROC                     glBindBufferRange;
PFNGLBINDRENDERBUFFERPROC                    glBindRenderbuffer;
PFNGLDELETERENDERBUFFERSPROC                 glDeleteRenderbuffers;
PFNGLGENRENDERBUFFERSPROC                    glGenRenderbuffers;
PFNGLINVALIDATEFRAMEBUFFERPROC               glInvalidateFramebuffer;
PFNGLBINDFRAMEBUFFERPROC                     glBindFramebuffer;
PFNGLDELETEFRAMEBUFFERSPROC                  glDeleteFramebuffers;
PFNGLGENFRAMEBUFFERSPROC                     glGenFramebuffers;
PFNGLFRAMEBUFFERTEXTURE2DPROC                glFramebufferTexture2D;
PFNGLFRAMEBUFFERRENDERBUFFERPROC             glFramebufferRenderbuffer;
PFNGLBLITFRAMEBUFFERPROC                     glBlitFramebuffer;
PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC      glRenderbufferStorageMultisample;
PFNGLBINDVERTEXARRAYPROC                     glBindVertexArray;
PFNGLDELETEVERTEXARRAYSPROC                  glDeleteVertexArrays;
PFNGLGENVERTEXARRAYSPROC                     glGenVertexArrays;
//PFNGLCOLORMASKIPROC                          glColorMaski;
//PFNGLGETBOOLEANI_VPROC                       glGetBooleani_v;
//PFNGLGETINTEGERI_VPROC                       glGetIntegeri_v;
//PFNGLENABLEIPROC                             glEnablei;
//PFNGLDISABLEIPROC                            glDisablei;
//PFNGLISENABLEDIPROC                          glIsEnabledi;
//PFNGLBEGINTRANSFORMFEEDBACKPROC              glBeginTransformFeedback;
//PFNGLENDTRANSFORMFEEDBACKPROC                glEndTransformFeedback;
//PFNGLBINDBUFFERBASEPROC                      glBindBufferBase;
//PFNGLTRANSFORMFEEDBACKVARYINGSPROC           glTransformFeedbackVaryings;
//PFNGLGETTRANSFORMFEEDBACKVARYINGPROC         glGetTransformFeedbackVarying;
//PFNGLCLAMPCOLORPROC                          glClampColor;
//PFNGLBEGINCONDITIONALRENDERPROC              glBeginConditionalRender;
//PFNGLENDCONDITIONALRENDERPROC                glEndConditionalRender;
//PFNGLVERTEXATTRIBIPOINTERPROC                glVertexAttribIPointer;
//PFNGLGETVERTEXATTRIBIIVPROC                  glGetVertexAttribIiv;
//PFNGLGETVERTEXATTRIBIUIVPROC                 glGetVertexAttribIuiv;
//PFNGLVERTEXATTRIBI1IPROC                     glVertexAttribI1i;
//PFNGLVERTEXATTRIBI2IPROC                     glVertexAttribI2i;
//PFNGLVERTEXATTRIBI3IPROC                     glVertexAttribI3i;
//PFNGLVERTEXATTRIBI4IPROC                     glVertexAttribI4i;
//PFNGLVERTEXATTRIBI1UIPROC                    glVertexAttribI1ui;
//PFNGLVERTEXATTRIBI2UIPROC                    glVertexAttribI2ui;
//PFNGLVERTEXATTRIBI3UIPROC                    glVertexAttribI3ui;
//PFNGLVERTEXATTRIBI4UIPROC                    glVertexAttribI4ui;
//PFNGLVERTEXATTRIBI1IVPROC                    glVertexAttribI1iv;
//PFNGLVERTEXATTRIBI2IVPROC                    glVertexAttribI2iv;
//PFNGLVERTEXATTRIBI3IVPROC                    glVertexAttribI3iv;
//PFNGLVERTEXATTRIBI4IVPROC                    glVertexAttribI4iv;
//PFNGLVERTEXATTRIBI1UIVPROC                   glVertexAttribI1uiv;
//PFNGLVERTEXATTRIBI2UIVPROC                   glVertexAttribI2uiv;
//PFNGLVERTEXATTRIBI3UIVPROC                   glVertexAttribI3uiv;
//PFNGLVERTEXATTRIBI4UIVPROC                   glVertexAttribI4uiv;
//PFNGLVERTEXATTRIBI4BVPROC                    glVertexAttribI4bv;
//PFNGLVERTEXATTRIBI4SVPROC                    glVertexAttribI4sv;
//PFNGLVERTEXATTRIBI4UBVPROC                   glVertexAttribI4ubv;
//PFNGLVERTEXATTRIBI4USVPROC                   glVertexAttribI4usv;
//PFNGLGETUNIFORMUIVPROC                       glGetUniformuiv;
//PFNGLBINDFRAGDATALOCATIONPROC                glBindFragDataLocation;
//PFNGLGETFRAGDATALOCATIONPROC                 glGetFragDataLocation;
//PFNGLUNIFORM1UIPROC                          glUniform1ui;
//PFNGLUNIFORM2UIPROC                          glUniform2ui;
//PFNGLUNIFORM3UIPROC                          glUniform3ui;
//PFNGLUNIFORM4UIPROC                          glUniform4ui;
//PFNGLUNIFORM1UIVPROC                         glUniform1uiv;
//PFNGLUNIFORM2UIVPROC                         glUniform2uiv;
//PFNGLUNIFORM3UIVPROC                         glUniform3uiv;
//PFNGLUNIFORM4UIVPROC                         glUniform4uiv;
//PFNGLTEXPARAMETERIIVPROC                     glTexParameterIiv;
//PFNGLTEXPARAMETERIUIVPROC                    glTexParameterIuiv;
//PFNGLGETTEXPARAMETERIIVPROC                  glGetTexParameterIiv;
//PFNGLGETTEXPARAMETERIUIVPROC                 glGetTexParameterIuiv;
//PFNGLCLEARBUFFERIVPROC                       glClearBufferiv;
//PFNGLCLEARBUFFERUIVPROC                      glClearBufferuiv;
//PFNGLCLEARBUFFERFVPROC                       glClearBufferfv;
//PFNGLCLEARBUFFERFIPROC                       glClearBufferfi;
//PFNGLGETSTRINGIPROC                          glGetStringi;
//PFNGLISRENDERBUFFERPROC                      glIsRenderbuffer;
//PFNGLRENDERBUFFERSTORAGEPROC                 glRenderbufferStorage;
//PFNGLGETRENDERBUFFERPARAMETERIVPROC          glGetRenderbufferParameteriv;
//PFNGLISFRAMEBUFFERPROC                       glIsFramebuffer;
//PFNGLCHECKFRAMEBUFFERSTATUSPROC              glCheckFramebufferStatus;
//PFNGLFRAMEBUFFERTEXTURE1DPROC                glFramebufferTexture1D;
//PFNGLFRAMEBUFFERTEXTURE3DPROC                glFramebufferTexture3D;
//PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC glGetFramebufferAttachmentParameteriv;
//PFNGLGENERATEMIPMAPPROC                      glGenerateMipmap;
//PFNGLFRAMEBUFFERTEXTURELAYERPROC             glFramebufferTextureLayer;
//PFNGLMAPBUFFERRANGEPROC                      glMapBufferRange;
//PFNGLFLUSHMAPPEDBUFFERRANGEPROC              glFlushMappedBufferRange;
//PFNGLISVERTEXARRAYPROC                       glIsVertexArray;

//GL_VERSION_3_1
PFNGLGETUNIFORMBLOCKINDEXPROC      glGetUniformBlockIndex;
PFNGLUNIFORMBLOCKBINDINGPROC       glUniformBlockBinding;
//PFNGLDRAWARRAYSINSTANCEDPROC       glDrawArraysInstanced;
//PFNGLDRAWELEMENTSINSTANCEDPROC     glDrawElementsInstanced;
//PFNGLTEXBUFFERPROC                 glTexBuffer;
//PFNGLPRIMITIVERESTARTINDEXPROC     glPrimitiveRestartIndex;
//PFNGLCOPYBUFFERSUBDATAPROC         glCopyBufferSubData;
//PFNGLGETUNIFORMINDICESPROC         glGetUniformIndices;
//PFNGLGETACTIVEUNIFORMSIVPROC       glGetActiveUniformsiv;
//PFNGLGETACTIVEUNIFORMNAMEPROC      glGetActiveUniformName;
//PFNGLGETACTIVEUNIFORMBLOCKIVPROC   glGetActiveUniformBlockiv;
//PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC glGetActiveUniformBlockName;

#if defined(_WIN32) && !defined(__CYGWIN__) && defined(THORVG_GL_TARGET_GL)
    PFNWGLGETCURRENTCONTEXTPROC  tvgWglGetCurrentContext;
    PFNWGLMAKECURRENTPROC        tvgWglMakeCurrent;
#endif

#if defined(THORVG_GL_TARGET_GLES)
    PFNEGLGETCURRENTCONTEXTPROC  tvgEglGetCurrentContext;
    PFNEGLMAKECURRENTPROC        tvgEglMakeCurrent;
#endif

bool glInit()
{
    if (!_glLoad()) return false;

    // GL_VERSION_1_0
    GL_FUNCTION_FETCH(glCullFace, PFNGLCULLFACEPROC);
    GL_FUNCTION_FETCH(glFrontFace, PFNGLFRONTFACEPROC);
    // GL_FUNCTION_FETCH(glHint, PFNGLHINTPROC);
    // GL_FUNCTION_FETCH(glLineWidth, PFNGLLINEWIDTHPROC);
    // GL_FUNCTION_FETCH(glPointSize, PFNGLPOINTSIZEPROC);
    // GL_FUNCTION_FETCH(glPolygonMode, PFNGLPOLYGONMODEPROC);
    GL_FUNCTION_FETCH(glScissor, PFNGLSCISSORPROC);
    // GL_FUNCTION_FETCH(glTexParameterf, PFNGLTEXPARAMETERFPROC);
    // GL_FUNCTION_FETCH(glTexParameterfv, PFNGLTEXPARAMETERFVPROC);
    GL_FUNCTION_FETCH(glTexParameteri, PFNGLTEXPARAMETERIPROC);
    // GL_FUNCTION_FETCH(glTexParameteriv, PFNGLTEXPARAMETERIVPROC);
    // GL_FUNCTION_FETCH(glTexImage1D, PFNGLTEXIMAGE1DPROC);
    GL_FUNCTION_FETCH(glTexImage2D, PFNGLTEXIMAGE2DPROC);
#if !defined(THORVG_GL_TARGET_GLES)
    GL_FUNCTION_FETCH(glDrawBuffer, PFNGLDRAWBUFFERPROC);
#endif
    GL_FUNCTION_FETCH(glClear, PFNGLCLEARPROC);
    GL_FUNCTION_FETCH(glClearColor, PFNGLCLEARCOLORPROC);
    GL_FUNCTION_FETCH(glClearStencil, PFNGLCLEARSTENCILPROC);
#if defined(THORVG_GL_TARGET_GLES)
    GL_FUNCTION_FETCH(glClearDepthf, PFNGLCLEARDEPTHFPROC);
#else
    GL_FUNCTION_FETCH(glClearDepth, PFNGLCLEARDEPTHPROC);
#endif
    // GL_FUNCTION_FETCH(glStencilMask, PFNGLSTENCILMASKPROC);
    GL_FUNCTION_FETCH(glColorMask, PFNGLCOLORMASKPROC);
    GL_FUNCTION_FETCH(glDepthMask, PFNGLDEPTHMASKPROC);
    GL_FUNCTION_FETCH(glDisable, PFNGLDISABLEPROC);
    GL_FUNCTION_FETCH(glEnable, PFNGLENABLEPROC);
    // GL_FUNCTION_FETCH(glFinish, PFNGLFINISHPROC);
    // GL_FUNCTION_FETCH(glFlush, PFNGLFLUSHPROC);
    GL_FUNCTION_FETCH(glBlendFunc, PFNGLBLENDFUNCPROC);
    // GL_FUNCTION_FETCH(glLogicOp, PFNGLLOGICOPPROC);
    GL_FUNCTION_FETCH(glStencilFunc, PFNGLSTENCILFUNCPROC);
    GL_FUNCTION_FETCH(glStencilOp, PFNGLSTENCILOPPROC);
    GL_FUNCTION_FETCH(glDepthFunc, PFNGLDEPTHFUNCPROC);
    // GL_FUNCTION_FETCH(glPixelStoref, PFNGLPIXELSTOREFPROC);
    // GL_FUNCTION_FETCH(glPixelStorei, PFNGLPIXELSTOREIPROC);
    // GL_FUNCTION_FETCH(glReadBuffer, PFNGLREADBUFFERPROC);
    // GL_FUNCTION_FETCH(glReadPixels, PFNGLREADPIXELSPROC);
    // GL_FUNCTION_FETCH(glGetBooleanv, PFNGLGETBOOLEANVPROC);
    // GL_FUNCTION_FETCH(glGetDoublev, PFNGLGETDOUBLEVPROC);
    GL_FUNCTION_FETCH(glGetError, PFNGLGETERRORPROC);
    // GL_FUNCTION_FETCH(glGetFloatv, PFNGLGETFLOATVPROC);
    GL_FUNCTION_FETCH(glGetIntegerv, PFNGLGETINTEGERVPROC);
    GL_FUNCTION_FETCH(glGetString, PFNGLGETSTRINGPROC);
    // GL_FUNCTION_FETCH(glGetTexImage, PFNGLGETTEXIMAGEPROC);
    // GL_FUNCTION_FETCH(glGetTexParameterfv, PFNGLGETTEXPARAMETERFVPROC);
    // GL_FUNCTION_FETCH(glGetTexParameteriv, PFNGLGETTEXPARAMETERIVPROC);
    // GL_FUNCTION_FETCH(glGetTexLevelParameterfv, PFNGLGETTEXLEVELPARAMETERFVPROC);
    // GL_FUNCTION_FETCH(glGetTexLevelParameteriv, PFNGLGETTEXLEVELPARAMETERIVPROC);
    // GL_FUNCTION_FETCH(glIsEnabled, PFNGLISENABLEDPROC);
    // GL_FUNCTION_FETCH(glDepthRange, PFNGLDEPTHRANGEPROC);
    GL_FUNCTION_FETCH(glViewport, PFNGLVIEWPORTPROC);

    // GL_VERSION_1_1
    // GL_FUNCTION_FETCH(glDrawArrays, PFNGLDRAWARRAYSPROC);
    GL_FUNCTION_FETCH(glDrawElements, PFNGLDRAWELEMENTSPROC);
    // GL_FUNCTION_FETCH(glGetPointerv, PFNGLGETPOINTERVPROC);
    // GL_FUNCTION_FETCH(glPolygonOffset, PFNGLPOLYGONOFFSETPROC);
    // GL_FUNCTION_FETCH(glCopyTexImage1D, PFNGLCOPYTEXIMAGE1DPROC);
    // GL_FUNCTION_FETCH(glCopyTexImage2D, PFNGLCOPYTEXIMAGE2DPROC);
    // GL_FUNCTION_FETCH(glCopyTexSubImage1D, PFNGLCOPYTEXSUBIMAGE1DPROC);
    // GL_FUNCTION_FETCH(glCopyTexSubImage2D, PFNGLCOPYTEXSUBIMAGE2DPROC);
    // GL_FUNCTION_FETCH(glTexSubImage1D, PFNGLTEXSUBIMAGE1DPROC);
    // GL_FUNCTION_FETCH(glTexSubImage2D, PFNGLTEXSUBIMAGE2DPROC);
    GL_FUNCTION_FETCH(glBindTexture, PFNGLBINDTEXTUREPROC);
    GL_FUNCTION_FETCH(glDeleteTextures, PFNGLDELETETEXTURESPROC);
    GL_FUNCTION_FETCH(glGenTextures, PFNGLGENTEXTURESPROC);
    // GL_FUNCTION_FETCH(glIsTexture, PFNGLISTEXTUREPROC);

    // // GL_VERSION_1_2
    // GL_FUNCTION_FETCH(glDrawRangeElements, PFNGLDRAWRANGEELEMENTSPROC);
    // GL_FUNCTION_FETCH(glTexImage3D, PFNGLTEXIMAGE3DPROC);
    // GL_FUNCTION_FETCH(glTexSubImage3D, PFNGLTEXSUBIMAGE3DPROC);
    // GL_FUNCTION_FETCH(glCopyTexSubImage3D, PFNGLCOPYTEXSUBIMAGE3DPROC);
    
    // // GL_VERSION_1_3
    GL_FUNCTION_FETCH(glActiveTexture, PFNGLACTIVETEXTUREPROC);
    // GL_FUNCTION_FETCH(glSampleCoverage, PFNGLSAMPLECOVERAGEPROC);
    // GL_FUNCTION_FETCH(glCompressedTexImage3D, PFNGLCOMPRESSEDTEXIMAGE3DPROC);
    // GL_FUNCTION_FETCH(glCompressedTexImage2D, PFNGLCOMPRESSEDTEXIMAGE2DPROC);
    // GL_FUNCTION_FETCH(glCompressedTexImage1D, PFNGLCOMPRESSEDTEXIMAGE1DPROC);
    // GL_FUNCTION_FETCH(glCompressedTexSubImage3D, PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC);
    // GL_FUNCTION_FETCH(glCompressedTexSubImage2D, PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC);
    // GL_FUNCTION_FETCH(glCompressedTexSubImage1D, PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC);
    // GL_FUNCTION_FETCH(glGetCompressedTexImage, PFNGLGETCOMPRESSEDTEXIMAGEPROC);
    
    // // GL_VERSION_1_4
    // GL_FUNCTION_FETCH(glBlendFuncSeparate, PFNGLBLENDFUNCSEPARATEPROC);
    // GL_FUNCTION_FETCH(glMultiDrawArrays, PFNGLMULTIDRAWARRAYSPROC);
    // GL_FUNCTION_FETCH(glMultiDrawElements, PFNGLMULTIDRAWELEMENTSPROC);
    // GL_FUNCTION_FETCH(glPointParameterf, PFNGLPOINTPARAMETERFPROC);
    // GL_FUNCTION_FETCH(glPointParameterfv, PFNGLPOINTPARAMETERFVPROC);
    // GL_FUNCTION_FETCH(glPointParameteri, PFNGLPOINTPARAMETERIPROC);
    // GL_FUNCTION_FETCH(glPointParameteriv, PFNGLPOINTPARAMETERIVPROC);
    // GL_FUNCTION_FETCH(glBlendColor, PFNGLBLENDCOLORPROC);
    GL_FUNCTION_FETCH(glBlendEquation, PFNGLBLENDEQUATIONPROC);

    // GL_VERSION_1_5
    // GL_FUNCTION_FETCH(glGenQueries, PFNGLGENQUERIESPROC);
    // GL_FUNCTION_FETCH(glDeleteQueries, PFNGLDELETEQUERIESPROC);
    // GL_FUNCTION_FETCH(glIsQuery, PFNGLISQUERYPROC);
    // GL_FUNCTION_FETCH(glBeginQuery, PFNGLBEGINQUERYPROC);
    // GL_FUNCTION_FETCH(glEndQuery, PFNGLENDQUERYPROC);
    // GL_FUNCTION_FETCH(glGetQueryiv, PFNGLGETQUERYIVPROC);
    // GL_FUNCTION_FETCH(glGetQueryObjectiv, PFNGLGETQUERYOBJECTIVPROC);
    // GL_FUNCTION_FETCH(glGetQueryObjectuiv, PFNGLGETQUERYOBJECTUIVPROC);
    GL_FUNCTION_FETCH(glBindBuffer, PFNGLBINDBUFFERPROC);
    GL_FUNCTION_FETCH(glDeleteBuffers, PFNGLDELETEBUFFERSPROC);
    GL_FUNCTION_FETCH(glGenBuffers, PFNGLGENBUFFERSPROC);
    // GL_FUNCTION_FETCH(glIsBuffer, PFNGLISBUFFERPROC);
    GL_FUNCTION_FETCH(glBufferData, PFNGLBUFFERDATAPROC);
    // GL_FUNCTION_FETCH(glBufferSubData, PFNGLBUFFERSUBDATAPROC);
    // GL_FUNCTION_FETCH(glGetBufferSubData, PFNGLGETBUFFERSUBDATAPROC);
    // GL_FUNCTION_FETCH(glMapBuffer, PFNGLMAPBUFFERPROC);
    // GL_FUNCTION_FETCH(glUnmapBuffer, PFNGLUNMAPBUFFERPROC);
    // GL_FUNCTION_FETCH(glGetBufferParameteriv, PFNGLGETBUFFERPARAMETERIVPROC);
    // GL_FUNCTION_FETCH(glGetBufferPointerv, PFNGLGETBUFFERPOINTERVPROC);

    // GL_VERSION_2_0
    // GL_FUNCTION_FETCH(glBlendEquationSeparate, PFNGLBLENDEQUATIONSEPARATEPROC);
#if defined(THORVG_GL_TARGET_GLES)
    GL_FUNCTION_FETCH(glDrawBuffers, PFNGLDRAWBUFFERSPROC);
#endif
    GL_FUNCTION_FETCH(glStencilOpSeparate, PFNGLSTENCILOPSEPARATEPROC);
    GL_FUNCTION_FETCH(glStencilFuncSeparate, PFNGLSTENCILFUNCSEPARATEPROC);
    // GL_FUNCTION_FETCH(glStencilMaskSeparate, PFNGLSTENCILMASKSEPARATEPROC);
    GL_FUNCTION_FETCH(glAttachShader, PFNGLATTACHSHADERPROC);
    // GL_FUNCTION_FETCH(glBindAttribLocation, PFNGLBINDATTRIBLOCATIONPROC);
    GL_FUNCTION_FETCH(glCompileShader, PFNGLCOMPILESHADERPROC);
    GL_FUNCTION_FETCH(glCreateProgram, PFNGLCREATEPROGRAMPROC);
    GL_FUNCTION_FETCH(glCreateShader, PFNGLCREATESHADERPROC);
    GL_FUNCTION_FETCH(glDeleteProgram, PFNGLDELETEPROGRAMPROC);
    GL_FUNCTION_FETCH(glDeleteShader, PFNGLDELETESHADERPROC);
    // GL_FUNCTION_FETCH(glDetachShader, PFNGLDETACHSHADERPROC);
    GL_FUNCTION_FETCH(glDisableVertexAttribArray, PFNGLDISABLEVERTEXATTRIBARRAYPROC);
    GL_FUNCTION_FETCH(glEnableVertexAttribArray, PFNGLENABLEVERTEXATTRIBARRAYPROC);
    // GL_FUNCTION_FETCH(glGetActiveAttrib, PFNGLGETACTIVEATTRIBPROC);
    // GL_FUNCTION_FETCH(glGetActiveUniform, PFNGLGETACTIVEUNIFORMPROC);
    // GL_FUNCTION_FETCH(glGetAttachedShaders, PFNGLGETATTACHEDSHADERSPROC);
    GL_FUNCTION_FETCH(glGetAttribLocation, PFNGLGETATTRIBLOCATIONPROC);
    GL_FUNCTION_FETCH(glGetProgramiv, PFNGLGETPROGRAMIVPROC);
    GL_FUNCTION_FETCH(glGetProgramInfoLog, PFNGLGETPROGRAMINFOLOGPROC);
    GL_FUNCTION_FETCH(glGetShaderiv, PFNGLGETSHADERIVPROC);
    GL_FUNCTION_FETCH(glGetShaderInfoLog, PFNGLGETSHADERINFOLOGPROC);
    // GL_FUNCTION_FETCH(glGetShaderSource, PFNGLGETSHADERSOURCEPROC);
    GL_FUNCTION_FETCH(glGetUniformLocation, PFNGLGETUNIFORMLOCATIONPROC);
    // GL_FUNCTION_FETCH(glGetUniformfv, PFNGLGETUNIFORMFVPROC);
    // GL_FUNCTION_FETCH(glGetUniformiv, PFNGLGETUNIFORMIVPROC);
    // GL_FUNCTION_FETCH(glGetVertexAttribdv, PFNGLGETVERTEXATTRIBDVPROC);
    // GL_FUNCTION_FETCH(glGetVertexAttribfv, PFNGLGETVERTEXATTRIBFVPROC);
    // GL_FUNCTION_FETCH(glGetVertexAttribiv, PFNGLGETVERTEXATTRIBIVPROC);
    // GL_FUNCTION_FETCH(glGetVertexAttribPointerv, PFNGLGETVERTEXATTRIBPOINTERVPROC);
    // GL_FUNCTION_FETCH(glIsProgram, PFNGLISPROGRAMPROC);
    // GL_FUNCTION_FETCH(glIsShader, PFNGLISSHADERPROC);
    GL_FUNCTION_FETCH(glLinkProgram, PFNGLLINKPROGRAMPROC);
    GL_FUNCTION_FETCH(glShaderSource, PFNGLSHADERSOURCEPROC);
    GL_FUNCTION_FETCH(glUseProgram, PFNGLUSEPROGRAMPROC);
    GL_FUNCTION_FETCH(glUniform1f, PFNGLUNIFORM1FPROC);
    // GL_FUNCTION_FETCH(glUniform2f, PFNGLUNIFORM2FPROC);
    // GL_FUNCTION_FETCH(glUniform3f, PFNGLUNIFORM3FPROC);
    // GL_FUNCTION_FETCH(glUniform4f, PFNGLUNIFORM4FPROC);
    // GL_FUNCTION_FETCH(glUniform1i, PFNGLUNIFORM1IPROC);
    // GL_FUNCTION_FETCH(glUniform2i, PFNGLUNIFORM2IPROC);
    // GL_FUNCTION_FETCH(glUniform3i, PFNGLUNIFORM3IPROC);
    // GL_FUNCTION_FETCH(glUniform4i, PFNGLUNIFORM4IPROC);
    GL_FUNCTION_FETCH(glUniform1fv, PFNGLUNIFORM1FVPROC);
    GL_FUNCTION_FETCH(glUniform2fv, PFNGLUNIFORM2FVPROC);
    GL_FUNCTION_FETCH(glUniform3fv, PFNGLUNIFORM3FVPROC);
    GL_FUNCTION_FETCH(glUniform4fv, PFNGLUNIFORM4FVPROC);
    GL_FUNCTION_FETCH(glUniform1iv, PFNGLUNIFORM1IVPROC);
    GL_FUNCTION_FETCH(glUniform2iv, PFNGLUNIFORM2IVPROC);
    GL_FUNCTION_FETCH(glUniform3iv, PFNGLUNIFORM3IVPROC);
    GL_FUNCTION_FETCH(glUniform4iv, PFNGLUNIFORM4IVPROC);
    // GL_FUNCTION_FETCH(glUniformMatrix2fv, PFNGLUNIFORMMATRIX2FVPROC);
    GL_FUNCTION_FETCH(glUniformMatrix3fv, PFNGLUNIFORMMATRIX3FVPROC);
    // GL_FUNCTION_FETCH(glUniformMatrix4fv, PFNGLUNIFORMMATRIX4FVPROC);
    // GL_FUNCTION_FETCH(glValidateProgram, PFNGLVALIDATEPROGRAMPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib1d, PFNGLVERTEXATTRIB1DPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib1dv, PFNGLVERTEXATTRIB1DVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib1f, PFNGLVERTEXATTRIB1FPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib1fv, PFNGLVERTEXATTRIB1FVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib1s, PFNGLVERTEXATTRIB1SPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib1sv, PFNGLVERTEXATTRIB1SVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib2d, PFNGLVERTEXATTRIB2DPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib2dv, PFNGLVERTEXATTRIB2DVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib2f, PFNGLVERTEXATTRIB2FPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib2fv, PFNGLVERTEXATTRIB2FVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib2s, PFNGLVERTEXATTRIB2SPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib2sv, PFNGLVERTEXATTRIB2SVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib3d, PFNGLVERTEXATTRIB3DPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib3dv, PFNGLVERTEXATTRIB3DVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib3f, PFNGLVERTEXATTRIB3FPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib3fv, PFNGLVERTEXATTRIB3FVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib3s, PFNGLVERTEXATTRIB3SPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib3sv, PFNGLVERTEXATTRIB3SVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4Nbv, PFNGLVERTEXATTRIB4NBVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4Niv, PFNGLVERTEXATTRIB4NIVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4Nsv, PFNGLVERTEXATTRIB4NSVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4Nub, PFNGLVERTEXATTRIB4NUBPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4Nubv, PFNGLVERTEXATTRIB4NUBVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4Nuiv, PFNGLVERTEXATTRIB4NUIVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4Nusv, PFNGLVERTEXATTRIB4NUSVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4bv, PFNGLVERTEXATTRIB4BVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4d, PFNGLVERTEXATTRIB4DPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4dv, PFNGLVERTEXATTRIB4DVPROC);
    GL_FUNCTION_FETCH(glVertexAttrib4f, PFNGLVERTEXATTRIB4FPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4fv, PFNGLVERTEXATTRIB4FVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4iv, PFNGLVERTEXATTRIB4IVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4s, PFNGLVERTEXATTRIB4SPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4sv, PFNGLVERTEXATTRIB4SVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4ubv, PFNGLVERTEXATTRIB4UBVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4uiv, PFNGLVERTEXATTRIB4UIVPROC);
    // GL_FUNCTION_FETCH(glVertexAttrib4usv, PFNGLVERTEXATTRIB4USVPROC);
    GL_FUNCTION_FETCH(glVertexAttribPointer, PFNGLVERTEXATTRIBPOINTERPROC);
    
    // // GL_VERSION_2_1
    // GL_FUNCTION_FETCH(glUniformMatrix2x3fv, PFNGLUNIFORMMATRIX2X3FVPROC);
    // GL_FUNCTION_FETCH(glUniformMatrix3x2fv, PFNGLUNIFORMMATRIX3X2FVPROC);
    // GL_FUNCTION_FETCH(glUniformMatrix2x4fv, PFNGLUNIFORMMATRIX2X4FVPROC);
    // GL_FUNCTION_FETCH(glUniformMatrix4x2fv, PFNGLUNIFORMMATRIX4X2FVPROC);
    // GL_FUNCTION_FETCH(glUniformMatrix3x4fv, PFNGLUNIFORMMATRIX3X4FVPROC);
    // GL_FUNCTION_FETCH(glUniformMatrix4x3fv, PFNGLUNIFORMMATRIX4X3FVPROC);
    
    // GL_VERSION_3_0
    // GL_FUNCTION_FETCH(glColorMaski, PFNGLCOLORMASKIPROC);
    // GL_FUNCTION_FETCH(glGetBooleani_v, PFNGLGETBOOLEANI_VPROC);
    // GL_FUNCTION_FETCH(glGetIntegeri_v, PFNGLGETINTEGERI_VPROC);
    // GL_FUNCTION_FETCH(glEnablei, PFNGLENABLEIPROC);
    // GL_FUNCTION_FETCH(glDisablei, PFNGLDISABLEIPROC);
    // GL_FUNCTION_FETCH(glIsEnabledi, PFNGLISENABLEDIPROC);
    // GL_FUNCTION_FETCH(glBeginTransformFeedback, PFNGLBEGINTRANSFORMFEEDBACKPROC);
    // GL_FUNCTION_FETCH(glEndTransformFeedback, PFNGLENDTRANSFORMFEEDBACKPROC);
    GL_FUNCTION_FETCH(glBindBufferRange, PFNGLBINDBUFFERRANGEPROC);
    // GL_FUNCTION_FETCH(glBindBufferBase, PFNGLBINDBUFFERBASEPROC);
    // GL_FUNCTION_FETCH(glTransformFeedbackVaryings, PFNGLTRANSFORMFEEDBACKVARYINGSPROC);
    // GL_FUNCTION_FETCH(glGetTransformFeedbackVarying, PFNGLGETTRANSFORMFEEDBACKVARYINGPROC);
    // GL_FUNCTION_FETCH(glClampColor, PFNGLCLAMPCOLORPROC);
    // GL_FUNCTION_FETCH(glBeginConditionalRender, PFNGLBEGINCONDITIONALRENDERPROC);
    // GL_FUNCTION_FETCH(glEndConditionalRender, PFNGLENDCONDITIONALRENDERPROC);
    // GL_FUNCTION_FETCH(glVertexAttribIPointer, PFNGLVERTEXATTRIBIPOINTERPROC);
    // GL_FUNCTION_FETCH(glGetVertexAttribIiv, PFNGLGETVERTEXATTRIBIIVPROC);
    // GL_FUNCTION_FETCH(glGetVertexAttribIuiv, PFNGLGETVERTEXATTRIBIUIVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI1i, PFNGLVERTEXATTRIBI1IPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI2i, PFNGLVERTEXATTRIBI2IPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI3i, PFNGLVERTEXATTRIBI3IPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI4i, PFNGLVERTEXATTRIBI4IPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI1ui, PFNGLVERTEXATTRIBI1UIPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI2ui, PFNGLVERTEXATTRIBI2UIPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI3ui, PFNGLVERTEXATTRIBI3UIPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI4ui, PFNGLVERTEXATTRIBI4UIPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI1iv, PFNGLVERTEXATTRIBI1IVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI2iv, PFNGLVERTEXATTRIBI2IVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI3iv, PFNGLVERTEXATTRIBI3IVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI4iv, PFNGLVERTEXATTRIBI4IVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI1uiv, PFNGLVERTEXATTRIBI1UIVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI2uiv, PFNGLVERTEXATTRIBI2UIVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI3uiv, PFNGLVERTEXATTRIBI3UIVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI4uiv, PFNGLVERTEXATTRIBI4UIVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI4bv, PFNGLVERTEXATTRIBI4BVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI4sv, PFNGLVERTEXATTRIBI4SVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI4ubv, PFNGLVERTEXATTRIBI4UBVPROC);
    // GL_FUNCTION_FETCH(glVertexAttribI4usv, PFNGLVERTEXATTRIBI4USVPROC);
    // GL_FUNCTION_FETCH(glGetUniformuiv, PFNGLGETUNIFORMUIVPROC);
    // GL_FUNCTION_FETCH(glBindFragDataLocation, PFNGLBINDFRAGDATALOCATIONPROC);
    // GL_FUNCTION_FETCH(glGetFragDataLocation, PFNGLGETFRAGDATALOCATIONPROC);
    // GL_FUNCTION_FETCH(glUniform1ui, PFNGLUNIFORM1UIPROC);
    // GL_FUNCTION_FETCH(glUniform2ui, PFNGLUNIFORM2UIPROC);
    // GL_FUNCTION_FETCH(glUniform3ui, PFNGLUNIFORM3UIPROC);
    // GL_FUNCTION_FETCH(glUniform4ui, PFNGLUNIFORM4UIPROC);
    // GL_FUNCTION_FETCH(glUniform1uiv, PFNGLUNIFORM1UIVPROC);
    // GL_FUNCTION_FETCH(glUniform2uiv, PFNGLUNIFORM2UIVPROC);
    // GL_FUNCTION_FETCH(glUniform3uiv, PFNGLUNIFORM3UIVPROC);
    // GL_FUNCTION_FETCH(glUniform4uiv, PFNGLUNIFORM4UIVPROC);
    // GL_FUNCTION_FETCH(glTexParameterIiv, PFNGLTEXPARAMETERIIVPROC);
    // GL_FUNCTION_FETCH(glTexParameterIuiv, PFNGLTEXPARAMETERIUIVPROC);
    // GL_FUNCTION_FETCH(glGetTexParameterIiv, PFNGLGETTEXPARAMETERIIVPROC);
    // GL_FUNCTION_FETCH(glGetTexParameterIuiv, PFNGLGETTEXPARAMETERIUIVPROC);
    // GL_FUNCTION_FETCH(glClearBufferiv, PFNGLCLEARBUFFERIVPROC);
    // GL_FUNCTION_FETCH(glClearBufferuiv, PFNGLCLEARBUFFERUIVPROC);
    // GL_FUNCTION_FETCH(glClearBufferfv, PFNGLCLEARBUFFERFVPROC);
    // GL_FUNCTION_FETCH(glClearBufferfi, PFNGLCLEARBUFFERFIPROC);
    // GL_FUNCTION_FETCH(glGetStringi, PFNGLGETSTRINGIPROC);
    // GL_FUNCTION_FETCH(glIsRenderbuffer, PFNGLISRENDERBUFFERPROC);
    GL_FUNCTION_FETCH(glBindRenderbuffer, PFNGLBINDRENDERBUFFERPROC);
    GL_FUNCTION_FETCH(glDeleteRenderbuffers, PFNGLDELETERENDERBUFFERSPROC);
    GL_FUNCTION_FETCH(glGenRenderbuffers, PFNGLGENRENDERBUFFERSPROC);
    // GL_FUNCTION_FETCH(glRenderbufferStorage, PFNGLRENDERBUFFERSTORAGEPROC);
    // GL_FUNCTION_FETCH(glGetRenderbufferParameteriv, PFNGLGETRENDERBUFFERPARAMETERIVPROC);
    // GL_FUNCTION_FETCH(glIsFramebuffer, PFNGLISFRAMEBUFFERPROC);
#if defined(THORVG_GL_TARGET_GLES)
    GL_FUNCTION_FETCH(glInvalidateFramebuffer, PFNGLINVALIDATEFRAMEBUFFERPROC);
#endif
    GL_FUNCTION_FETCH(glBindFramebuffer, PFNGLBINDFRAMEBUFFERPROC);
    GL_FUNCTION_FETCH(glDeleteFramebuffers, PFNGLDELETEFRAMEBUFFERSPROC);
    GL_FUNCTION_FETCH(glGenFramebuffers, PFNGLGENFRAMEBUFFERSPROC);
    // GL_FUNCTION_FETCH(glCheckFramebufferStatus, PFNGLCHECKFRAMEBUFFERSTATUSPROC);
    // GL_FUNCTION_FETCH(glFramebufferTexture1D, PFNGLFRAMEBUFFERTEXTURE1DPROC);
    GL_FUNCTION_FETCH(glFramebufferTexture2D, PFNGLFRAMEBUFFERTEXTURE2DPROC);
    // GL_FUNCTION_FETCH(glFramebufferTexture3D, PFNGLFRAMEBUFFERTEXTURE3DPROC);
    GL_FUNCTION_FETCH(glFramebufferRenderbuffer, PFNGLFRAMEBUFFERRENDERBUFFERPROC);
    // GL_FUNCTION_FETCH(glGetFramebufferAttachmentParameteriv, PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC);
    // GL_FUNCTION_FETCH(glGenerateMipmap, PFNGLGENERATEMIPMAPPROC);
    GL_FUNCTION_FETCH(glBlitFramebuffer, PFNGLBLITFRAMEBUFFERPROC);
    GL_FUNCTION_FETCH(glRenderbufferStorageMultisample, PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC);
    // GL_FUNCTION_FETCH(glFramebufferTextureLayer, PFNGLFRAMEBUFFERTEXTURELAYERPROC);
    // GL_FUNCTION_FETCH(glMapBufferRange, PFNGLMAPBUFFERRANGEPROC);
    // GL_FUNCTION_FETCH(glFlushMappedBufferRange, PFNGLFLUSHMAPPEDBUFFERRANGEPROC);
    GL_FUNCTION_FETCH(glBindVertexArray, PFNGLBINDVERTEXARRAYPROC);
    GL_FUNCTION_FETCH(glDeleteVertexArrays, PFNGLDELETEVERTEXARRAYSPROC);
    GL_FUNCTION_FETCH(glGenVertexArrays, PFNGLGENVERTEXARRAYSPROC);
    // GL_FUNCTION_FETCH(glIsVertexArray, PFNGLISVERTEXARRAYPROC);
    
    // GL_VERSION_3_1
    // GL_FUNCTION_FETCH(glDrawArraysInstanced, PFNGLDRAWARRAYSINSTANCEDPROC);
    // GL_FUNCTION_FETCH(glDrawElementsInstanced, PFNGLDRAWELEMENTSINSTANCEDPROC);
    // GL_FUNCTION_FETCH(glTexBuffer, PFNGLTEXBUFFERPROC);
    // GL_FUNCTION_FETCH(glPrimitiveRestartIndex, PFNGLPRIMITIVERESTARTINDEXPROC);
    // GL_FUNCTION_FETCH(glCopyBufferSubData, PFNGLCOPYBUFFERSUBDATAPROC);
    // GL_FUNCTION_FETCH(glGetUniformIndices, PFNGLGETUNIFORMINDICESPROC);
    // GL_FUNCTION_FETCH(glGetActiveUniformsiv, PFNGLGETACTIVEUNIFORMSIVPROC);
    // GL_FUNCTION_FETCH(glGetActiveUniformName, PFNGLGETACTIVEUNIFORMNAMEPROC);
    GL_FUNCTION_FETCH(glGetUniformBlockIndex, PFNGLGETUNIFORMBLOCKINDEXPROC);
    // GL_FUNCTION_FETCH(glGetActiveUniformBlockiv, PFNGLGETACTIVEUNIFORMBLOCKIVPROC);
    // GL_FUNCTION_FETCH(glGetActiveUniformBlockName, PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC);
    GL_FUNCTION_FETCH(glUniformBlockBinding, PFNGLUNIFORMBLOCKBINDINGPROC);

#if defined(_WIN32) && !defined(__CYGWIN__) && defined(THORVG_GL_TARGET_GL)
    tvgWglGetCurrentContext = (PFNWGLGETCURRENTCONTEXTPROC)GetProcAddress(_libGL, "wglGetCurrentContext");
    tvgWglMakeCurrent = (PFNWGLMAKECURRENTPROC)GetProcAddress(_libGL, "wglMakeCurrent");
    if (!tvgWglGetCurrentContext || !tvgWglMakeCurrent) {
        TVGERR("GL_ENGINE", "Failed to load WGL context management functions.");
        return false;
    }
#endif

#if defined(THORVG_GL_TARGET_GLES)
    #if defined(_WIN32) && !defined(__CYGWIN__)
        if (!_libEGL) _libEGL = LoadLibraryW(L"libEGL.dll");
        if (!_libEGL) _libEGL = LoadLibraryW(L"EGL.dll");
        if (!_libEGL) {
            TVGERR("GL_ENGINE", "Cannot find EGL library.");
            return false;
        }
        tvgEglGetCurrentContext = (PFNEGLGETCURRENTCONTEXTPROC)GetProcAddress(_libEGL, "eglGetCurrentContext");
        tvgEglMakeCurrent = (PFNEGLMAKECURRENTPROC)GetProcAddress(_libEGL, "eglMakeCurrent");
    #else
        if (!_libEGL) _libEGL = dlopen("libEGL.so.1", RTLD_LAZY);
        if (!_libEGL) _libEGL = dlopen("libEGL.so", RTLD_LAZY);
        if (!_libEGL) {
            TVGERR("GL_ENGINE", "Cannot find EGL library.");
            return false;
        }
        tvgEglGetCurrentContext = (PFNEGLGETCURRENTCONTEXTPROC)dlsym(_libEGL, "eglGetCurrentContext");
        tvgEglMakeCurrent = (PFNEGLMAKECURRENTPROC)dlsym(_libEGL, "eglMakeCurrent");
    #endif
    if (!tvgEglGetCurrentContext || !tvgEglMakeCurrent) {
        TVGERR("GL_ENGINE", "Failed to load EGL context management functions.");
        return false;
    }
#endif

    //Confirm the version
    GLint vMajor, vMinor;
    glGetIntegerv(GL_MAJOR_VERSION, &vMajor);
    glGetIntegerv(GL_MINOR_VERSION, &vMinor);
    if (vMajor < TVG_REQUIRE_GL_MAJOR_VER || (vMajor ==  TVG_REQUIRE_GL_MAJOR_VER && vMinor <  TVG_REQUIRE_GL_MINOR_VER)) {
        TVGERR("GL_ENGINE", "OpenGL/ES version is not satisfied. Current: v%d.%d, Required: v%d.%d", vMajor, vMinor, TVG_REQUIRE_GL_MAJOR_VER, TVG_REQUIRE_GL_MINOR_VER);
        return false;
    }

    TVGLOG("GL_ENGINE", "OpenGL/ES version = v%d.%d", vMajor, vMinor);

    return true;
};


bool glTerm()
{
#if defined(_WIN32) && !defined(__CYGWIN__)
    if (_libGL) FreeLibrary(_libGL);
    #if defined(THORVG_GL_TARGET_GLES)
        if (_libEGL) FreeLibrary(_libEGL);
    #endif
#else
    if (_libGL) dlclose(_libGL);
    #if defined(THORVG_GL_TARGET_GLES)
        if (_libEGL) dlclose(_libEGL);
    #endif
#endif

    return true;
}

#endif // __EMSCRIPTEN__
