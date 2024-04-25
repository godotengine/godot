//========================================================================
// GLFW 3.4 - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2016 Google Inc.
// Copyright (c) 2016-2019 Camilla Löwy <elmindreda@glfw.org>
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
// It is fine to use C99 in this file because it will not be built with VS
//========================================================================

#include "internal.h"


static int createNativeWindow(_GLFWwindow* window,
                              const _GLFWwndconfig* wndconfig)
{
    window->null.width = wndconfig->width;
    window->null.height = wndconfig->height;

    return GLFW_TRUE;
}


//////////////////////////////////////////////////////////////////////////
//////                       GLFW platform API                      //////
//////////////////////////////////////////////////////////////////////////

int _glfwPlatformCreateWindow(_GLFWwindow* window,
                              const _GLFWwndconfig* wndconfig,
                              const _GLFWctxconfig* ctxconfig,
                              const _GLFWfbconfig* fbconfig)
{
    if (!createNativeWindow(window, wndconfig))
        return GLFW_FALSE;

    if (ctxconfig->client != GLFW_NO_API)
    {
        if (ctxconfig->source == GLFW_NATIVE_CONTEXT_API ||
            ctxconfig->source == GLFW_OSMESA_CONTEXT_API)
        {
            if (!_glfwInitOSMesa())
                return GLFW_FALSE;
            if (!_glfwCreateContextOSMesa(window, ctxconfig, fbconfig))
                return GLFW_FALSE;
        }
        else
        {
            _glfwInputError(GLFW_API_UNAVAILABLE, "Null: EGL not available");
            return GLFW_FALSE;
        }
    }

    return GLFW_TRUE;
}

void _glfwPlatformDestroyWindow(_GLFWwindow* window)
{
    if (window->context.destroy)
        window->context.destroy(window);
}

void _glfwPlatformSetWindowTitle(_GLFWwindow* window, const char* title)
{
}

void _glfwPlatformSetWindowIcon(_GLFWwindow* window, int count,
                                const GLFWimage* images)
{
}

void _glfwPlatformSetWindowMonitor(_GLFWwindow* window,
                                   _GLFWmonitor* monitor,
                                   int xpos, int ypos,
                                   int width, int height,
                                   int refreshRate)
{
}

void _glfwPlatformGetWindowPos(_GLFWwindow* window, int* xpos, int* ypos)
{
}

void _glfwPlatformSetWindowPos(_GLFWwindow* window, int xpos, int ypos)
{
}

void _glfwPlatformGetWindowSize(_GLFWwindow* window, int* width, int* height)
{
    if (width)
        *width = window->null.width;
    if (height)
        *height = window->null.height;
}

void _glfwPlatformSetWindowSize(_GLFWwindow* window, int width, int height)
{
    window->null.width = width;
    window->null.height = height;
}

void _glfwPlatformSetWindowSizeLimits(_GLFWwindow* window,
                                      int minwidth, int minheight,
                                      int maxwidth, int maxheight)
{
}

void _glfwPlatformSetWindowAspectRatio(_GLFWwindow* window, int n, int d)
{
}

void _glfwPlatformGetFramebufferSize(_GLFWwindow* window, int* width, int* height)
{
    if (width)
        *width = window->null.width;
    if (height)
        *height = window->null.height;
}

void _glfwPlatformGetWindowFrameSize(_GLFWwindow* window,
                                     int* left, int* top,
                                     int* right, int* bottom)
{
}

void _glfwPlatformGetWindowContentScale(_GLFWwindow* window,
                                        float* xscale, float* yscale)
{
    if (xscale)
        *xscale = 1.f;
    if (yscale)
        *yscale = 1.f;
}

void _glfwPlatformIconifyWindow(_GLFWwindow* window)
{
}

void _glfwPlatformRestoreWindow(_GLFWwindow* window)
{
}

void _glfwPlatformMaximizeWindow(_GLFWwindow* window)
{
}

int _glfwPlatformWindowMaximized(_GLFWwindow* window)
{
    return GLFW_FALSE;
}

int _glfwPlatformWindowHovered(_GLFWwindow* window)
{
    return GLFW_FALSE;
}

int _glfwPlatformFramebufferTransparent(_GLFWwindow* window)
{
    return GLFW_FALSE;
}

void _glfwPlatformSetWindowResizable(_GLFWwindow* window, GLFWbool enabled)
{
}

void _glfwPlatformSetWindowDecorated(_GLFWwindow* window, GLFWbool enabled)
{
}

void _glfwPlatformSetWindowFloating(_GLFWwindow* window, GLFWbool enabled)
{
}

float _glfwPlatformGetWindowOpacity(_GLFWwindow* window)
{
    return 1.f;
}

void _glfwPlatformSetWindowOpacity(_GLFWwindow* window, float opacity)
{
}

void _glfwPlatformSetRawMouseMotion(_GLFWwindow *window, GLFWbool enabled)
{
}

GLFWbool _glfwPlatformRawMouseMotionSupported(void)
{
    return GLFW_FALSE;
}

void _glfwPlatformShowWindow(_GLFWwindow* window)
{
}


void _glfwPlatformRequestWindowAttention(_GLFWwindow* window)
{
}

void _glfwPlatformUnhideWindow(_GLFWwindow* window)
{
}

void _glfwPlatformHideWindow(_GLFWwindow* window)
{
}

void _glfwPlatformFocusWindow(_GLFWwindow* window)
{
}

int _glfwPlatformWindowFocused(_GLFWwindow* window)
{
    return GLFW_FALSE;
}

int _glfwPlatformWindowIconified(_GLFWwindow* window)
{
    return GLFW_FALSE;
}

int _glfwPlatformWindowVisible(_GLFWwindow* window)
{
    return GLFW_FALSE;
}

void _glfwPlatformPollEvents(void)
{
}

void _glfwPlatformWaitEvents(void)
{
}

void _glfwPlatformWaitEventsTimeout(double timeout)
{
}

void _glfwPlatformPostEmptyEvent(void)
{
}

void _glfwPlatformGetCursorPos(_GLFWwindow* window, double* xpos, double* ypos)
{
}

void _glfwPlatformSetCursorPos(_GLFWwindow* window, double x, double y)
{
}

void _glfwPlatformSetCursorMode(_GLFWwindow* window, int mode)
{
}

int _glfwPlatformCreateCursor(_GLFWcursor* cursor,
                              const GLFWimage* image,
                              int xhot, int yhot)
{
    return GLFW_TRUE;
}

int _glfwPlatformCreateStandardCursor(_GLFWcursor* cursor, int shape)
{
    return GLFW_TRUE;
}

void _glfwPlatformDestroyCursor(_GLFWcursor* cursor)
{
}

void _glfwPlatformSetCursor(_GLFWwindow* window, _GLFWcursor* cursor)
{
}

void _glfwPlatformSetClipboardString(const char* string)
{
}

const char* _glfwPlatformGetClipboardString(void)
{
    return NULL;
}

const char* _glfwPlatformGetScancodeName(int scancode)
{
    return "";
}

int _glfwPlatformGetKeyScancode(int key)
{
    return -1;
}

void _glfwPlatformGetRequiredInstanceExtensions(char** extensions)
{
}

int _glfwPlatformGetPhysicalDevicePresentationSupport(VkInstance instance,
                                                      VkPhysicalDevice device,
                                                      uint32_t queuefamily)
{
    return GLFW_FALSE;
}

VkResult _glfwPlatformCreateWindowSurface(VkInstance instance,
                                          _GLFWwindow* window,
                                          const VkAllocationCallbacks* allocator,
                                          VkSurfaceKHR* surface)
{
    // This seems like the most appropriate error to return here
    return VK_ERROR_INITIALIZATION_FAILED;
}

