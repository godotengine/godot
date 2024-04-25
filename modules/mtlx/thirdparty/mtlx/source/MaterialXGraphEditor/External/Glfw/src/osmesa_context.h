//========================================================================
// GLFW 3.4 OSMesa - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2016 Google Inc.
// Copyright (c) 2016-2017 Camilla Löwy <elmindreda@glfw.org>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================

#define OSMESA_RGBA 0x1908
#define OSMESA_FORMAT 0x22
#define OSMESA_DEPTH_BITS 0x30
#define OSMESA_STENCIL_BITS 0x31
#define OSMESA_ACCUM_BITS 0x32
#define OSMESA_PROFILE 0x33
#define OSMESA_CORE_PROFILE 0x34
#define OSMESA_COMPAT_PROFILE 0x35
#define OSMESA_CONTEXT_MAJOR_VERSION 0x36
#define OSMESA_CONTEXT_MINOR_VERSION 0x37

typedef void* OSMesaContext;
typedef void (*OSMESAproc)(void);

typedef OSMesaContext (GLAPIENTRY * PFN_OSMesaCreateContextExt)(GLenum,GLint,GLint,GLint,OSMesaContext);
typedef OSMesaContext (GLAPIENTRY * PFN_OSMesaCreateContextAttribs)(const int*,OSMesaContext);
typedef void (GLAPIENTRY * PFN_OSMesaDestroyContext)(OSMesaContext);
typedef int (GLAPIENTRY * PFN_OSMesaMakeCurrent)(OSMesaContext,void*,int,int,int);
typedef int (GLAPIENTRY * PFN_OSMesaGetColorBuffer)(OSMesaContext,int*,int*,int*,void**);
typedef int (GLAPIENTRY * PFN_OSMesaGetDepthBuffer)(OSMesaContext,int*,int*,int*,void**);
typedef GLFWglproc (GLAPIENTRY * PFN_OSMesaGetProcAddress)(const char*);
#define OSMesaCreateContextExt _glfw.osmesa.CreateContextExt
#define OSMesaCreateContextAttribs _glfw.osmesa.CreateContextAttribs
#define OSMesaDestroyContext _glfw.osmesa.DestroyContext
#define OSMesaMakeCurrent _glfw.osmesa.MakeCurrent
#define OSMesaGetColorBuffer _glfw.osmesa.GetColorBuffer
#define OSMesaGetDepthBuffer _glfw.osmesa.GetDepthBuffer
#define OSMesaGetProcAddress _glfw.osmesa.GetProcAddress

#define _GLFW_OSMESA_CONTEXT_STATE              _GLFWcontextOSMesa osmesa
#define _GLFW_OSMESA_LIBRARY_CONTEXT_STATE      _GLFWlibraryOSMesa osmesa


// OSMesa-specific per-context data
//
typedef struct _GLFWcontextOSMesa
{
    OSMesaContext       handle;
    int                 width;
    int                 height;
    void*               buffer;

} _GLFWcontextOSMesa;

// OSMesa-specific global data
//
typedef struct _GLFWlibraryOSMesa
{
    void*           handle;

    PFN_OSMesaCreateContextExt      CreateContextExt;
    PFN_OSMesaCreateContextAttribs  CreateContextAttribs;
    PFN_OSMesaDestroyContext        DestroyContext;
    PFN_OSMesaMakeCurrent           MakeCurrent;
    PFN_OSMesaGetColorBuffer        GetColorBuffer;
    PFN_OSMesaGetDepthBuffer        GetDepthBuffer;
    PFN_OSMesaGetProcAddress        GetProcAddress;

} _GLFWlibraryOSMesa;


GLFWbool _glfwInitOSMesa(void);
void _glfwTerminateOSMesa(void);
GLFWbool _glfwCreateContextOSMesa(_GLFWwindow* window,
                                  const _GLFWctxconfig* ctxconfig,
                                  const _GLFWfbconfig* fbconfig);

