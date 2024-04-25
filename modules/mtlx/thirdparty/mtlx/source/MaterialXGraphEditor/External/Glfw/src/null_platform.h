//========================================================================
// GLFW 3.4 - www.glfw.org
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

#include <dlfcn.h>

#define _GLFW_PLATFORM_WINDOW_STATE _GLFWwindowNull null

#define _GLFW_PLATFORM_CONTEXT_STATE         struct { int dummyContext; }
#define _GLFW_PLATFORM_MONITOR_STATE         struct { int dummyMonitor; }
#define _GLFW_PLATFORM_CURSOR_STATE          struct { int dummyCursor; }
#define _GLFW_PLATFORM_LIBRARY_WINDOW_STATE  struct { int dummyLibraryWindow; }
#define _GLFW_PLATFORM_LIBRARY_CONTEXT_STATE struct { int dummyLibraryContext; }
#define _GLFW_EGL_CONTEXT_STATE              struct { int dummyEGLContext; }
#define _GLFW_EGL_LIBRARY_CONTEXT_STATE      struct { int dummyEGLLibraryContext; }

#include "osmesa_context.h"
#include "posix_time.h"
#include "posix_thread.h"
#include "null_joystick.h"

#if defined(_GLFW_WIN32)
 #define _glfw_dlopen(name) LoadLibraryA(name)
 #define _glfw_dlclose(handle) FreeLibrary((HMODULE) handle)
 #define _glfw_dlsym(handle, name) GetProcAddress((HMODULE) handle, name)
#else
 #define _glfw_dlopen(name) dlopen(name, RTLD_LAZY | RTLD_LOCAL)
 #define _glfw_dlclose(handle) dlclose(handle)
 #define _glfw_dlsym(handle, name) dlsym(handle, name)
#endif

// Null-specific per-window data
//
typedef struct _GLFWwindowNull
{
    int width;
    int height;
} _GLFWwindowNull;

