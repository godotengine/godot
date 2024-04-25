//========================================================================
// GLFW 3.4 Wayland - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2014 Jonas Ådahl <jadahl@gmail.com>
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>


static void outputHandleGeometry(void* data,
                                 struct wl_output* output,
                                 int32_t x,
                                 int32_t y,
                                 int32_t physicalWidth,
                                 int32_t physicalHeight,
                                 int32_t subpixel,
                                 const char* make,
                                 const char* model,
                                 int32_t transform)
{
    struct _GLFWmonitor *monitor = data;
    char name[1024];

    monitor->wl.x = x;
    monitor->wl.y = y;
    monitor->widthMM = physicalWidth;
    monitor->heightMM = physicalHeight;

    snprintf(name, sizeof(name), "%s %s", make, model);
    monitor->name = _glfw_strdup(name);
}

static void outputHandleMode(void* data,
                             struct wl_output* output,
                             uint32_t flags,
                             int32_t width,
                             int32_t height,
                             int32_t refresh)
{
    struct _GLFWmonitor *monitor = data;
    GLFWvidmode mode;

    mode.width = width;
    mode.height = height;
    mode.redBits = 8;
    mode.greenBits = 8;
    mode.blueBits = 8;
    mode.refreshRate = (int) round(refresh / 1000.0);

    monitor->modeCount++;
    monitor->modes =
        realloc(monitor->modes, monitor->modeCount * sizeof(GLFWvidmode));
    monitor->modes[monitor->modeCount - 1] = mode;

    if (flags & WL_OUTPUT_MODE_CURRENT)
        monitor->wl.currentMode = monitor->modeCount - 1;
}

static void outputHandleDone(void* data, struct wl_output* output)
{
    struct _GLFWmonitor *monitor = data;

    _glfwInputMonitor(monitor, GLFW_CONNECTED, _GLFW_INSERT_LAST);
}

static void outputHandleScale(void* data,
                              struct wl_output* output,
                              int32_t factor)
{
    struct _GLFWmonitor *monitor = data;

    monitor->wl.scale = factor;
}

static const struct wl_output_listener outputListener = {
    outputHandleGeometry,
    outputHandleMode,
    outputHandleDone,
    outputHandleScale,
};


//////////////////////////////////////////////////////////////////////////
//////                       GLFW internal API                      //////
//////////////////////////////////////////////////////////////////////////

void _glfwAddOutputWayland(uint32_t name, uint32_t version)
{
    _GLFWmonitor *monitor;
    struct wl_output *output;

    if (version < 2)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Unsupported output interface version");
        return;
    }

    // The actual name of this output will be set in the geometry handler.
    monitor = _glfwAllocMonitor(NULL, 0, 0);

    output = wl_registry_bind(_glfw.wl.registry,
                              name,
                              &wl_output_interface,
                              2);
    if (!output)
    {
        _glfwFreeMonitor(monitor);
        return;
    }

    monitor->wl.scale = 1;
    monitor->wl.output = output;
    monitor->wl.name = name;

    wl_output_add_listener(output, &outputListener, monitor);
}


//////////////////////////////////////////////////////////////////////////
//////                       GLFW platform API                      //////
//////////////////////////////////////////////////////////////////////////

void _glfwPlatformFreeMonitor(_GLFWmonitor* monitor)
{
    if (monitor->wl.output)
        wl_output_destroy(monitor->wl.output);
}

void _glfwPlatformGetMonitorPos(_GLFWmonitor* monitor, int* xpos, int* ypos)
{
    if (xpos)
        *xpos = monitor->wl.x;
    if (ypos)
        *ypos = monitor->wl.y;
}

void _glfwPlatformGetMonitorContentScale(_GLFWmonitor* monitor,
                                         float* xscale, float* yscale)
{
    if (xscale)
        *xscale = (float) monitor->wl.scale;
    if (yscale)
        *yscale = (float) monitor->wl.scale;
}

void _glfwPlatformGetMonitorWorkarea(_GLFWmonitor* monitor,
                                     int* xpos, int* ypos,
                                     int* width, int* height)
{
    if (xpos)
        *xpos = monitor->wl.x;
    if (ypos)
        *ypos = monitor->wl.y;
    if (width)
        *width = monitor->modes[monitor->wl.currentMode].width;
    if (height)
        *height = monitor->modes[monitor->wl.currentMode].height;
}

GLFWvidmode* _glfwPlatformGetVideoModes(_GLFWmonitor* monitor, int* found)
{
    *found = monitor->modeCount;
    return monitor->modes;
}

void _glfwPlatformGetVideoMode(_GLFWmonitor* monitor, GLFWvidmode* mode)
{
    *mode = monitor->modes[monitor->wl.currentMode];
}

GLFWbool _glfwPlatformGetGammaRamp(_GLFWmonitor* monitor, GLFWgammaramp* ramp)
{
    _glfwInputError(GLFW_PLATFORM_ERROR,
                    "Wayland: Gamma ramp access is not available");
    return GLFW_FALSE;
}

void _glfwPlatformSetGammaRamp(_GLFWmonitor* monitor,
                               const GLFWgammaramp* ramp)
{
    _glfwInputError(GLFW_PLATFORM_ERROR,
                    "Wayland: Gamma ramp access is not available");
}


//////////////////////////////////////////////////////////////////////////
//////                        GLFW native API                       //////
//////////////////////////////////////////////////////////////////////////

GLFWAPI struct wl_output* glfwGetWaylandMonitor(GLFWmonitor* handle)
{
    _GLFWmonitor* monitor = (_GLFWmonitor*) handle;
    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    return monitor->wl.output;
}

