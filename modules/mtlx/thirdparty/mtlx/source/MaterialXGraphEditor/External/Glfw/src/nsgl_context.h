//========================================================================
// GLFW 3.4 macOS - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2009-2019 Camilla Löwy <elmindreda@glfw.org>
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

// NOTE: Many Cocoa enum values have been renamed and we need to build across
//       SDK versions where one is unavailable or the other deprecated
//       We use the newer names in code and these macros to handle compatibility
#if MAC_OS_X_VERSION_MAX_ALLOWED < 101400
 #define NSOpenGLContextParameterSwapInterval NSOpenGLCPSwapInterval
 #define NSOpenGLContextParameterSurfaceOpacity NSOpenGLCPSurfaceOpacity
#endif

#define _GLFW_PLATFORM_CONTEXT_STATE            _GLFWcontextNSGL nsgl
#define _GLFW_PLATFORM_LIBRARY_CONTEXT_STATE    _GLFWlibraryNSGL nsgl

#include <stdatomic.h>


// NSGL-specific per-context data
//
typedef struct _GLFWcontextNSGL
{
    id                pixelFormat;
    id                object;

} _GLFWcontextNSGL;

// NSGL-specific global data
//
typedef struct _GLFWlibraryNSGL
{
    // dlopen handle for OpenGL.framework (for glfwGetProcAddress)
    CFBundleRef     framework;

} _GLFWlibraryNSGL;


GLFWbool _glfwInitNSGL(void);
void _glfwTerminateNSGL(void);
GLFWbool _glfwCreateContextNSGL(_GLFWwindow* window,
                                const _GLFWctxconfig* ctxconfig,
                                const _GLFWfbconfig* fbconfig);
void _glfwDestroyContextNSGL(_GLFWwindow* window);

