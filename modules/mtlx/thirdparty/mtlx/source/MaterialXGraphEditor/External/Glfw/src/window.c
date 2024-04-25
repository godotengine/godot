//========================================================================
// GLFW 3.4 - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2002-2006 Marcus Geelnard
// Copyright (c) 2006-2019 Camilla Löwy <elmindreda@glfw.org>
// Copyright (c) 2012 Torsten Walluhn <tw@mad-cad.net>
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
// Please use C89 style variable declarations in this file because VS 2010
//========================================================================

#include "internal.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>


//////////////////////////////////////////////////////////////////////////
//////                         GLFW event API                       //////
//////////////////////////////////////////////////////////////////////////

// Notifies shared code that a window has lost or received input focus
//
void _glfwInputWindowFocus(_GLFWwindow* window, GLFWbool focused)
{
    if (window->callbacks.focus)
        window->callbacks.focus((GLFWwindow*) window, focused);

    if (!focused)
    {
        int key, button;

        for (key = 0;  key <= GLFW_KEY_LAST;  key++)
        {
            if (window->keys[key] == GLFW_PRESS)
            {
                const int scancode = _glfwPlatformGetKeyScancode(key);
                _glfwInputKey(window, key, scancode, GLFW_RELEASE, 0);
            }
        }

        for (button = 0;  button <= GLFW_MOUSE_BUTTON_LAST;  button++)
        {
            if (window->mouseButtons[button] == GLFW_PRESS)
                _glfwInputMouseClick(window, button, GLFW_RELEASE, 0);
        }
    }
}

// Notifies shared code that a window has moved
// The position is specified in content area relative screen coordinates
//
void _glfwInputWindowPos(_GLFWwindow* window, int x, int y)
{
    if (window->callbacks.pos)
        window->callbacks.pos((GLFWwindow*) window, x, y);
}

// Notifies shared code that a window has been resized
// The size is specified in screen coordinates
//
void _glfwInputWindowSize(_GLFWwindow* window, int width, int height)
{
    if (window->callbacks.size)
        window->callbacks.size((GLFWwindow*) window, width, height);
}

// Notifies shared code that a window has been iconified or restored
//
void _glfwInputWindowIconify(_GLFWwindow* window, GLFWbool iconified)
{
    if (window->callbacks.iconify)
        window->callbacks.iconify((GLFWwindow*) window, iconified);
}

// Notifies shared code that a window has been maximized or restored
//
void _glfwInputWindowMaximize(_GLFWwindow* window, GLFWbool maximized)
{
    if (window->callbacks.maximize)
        window->callbacks.maximize((GLFWwindow*) window, maximized);
}

// Notifies shared code that a window framebuffer has been resized
// The size is specified in pixels
//
void _glfwInputFramebufferSize(_GLFWwindow* window, int width, int height)
{
    if (window->callbacks.fbsize)
        window->callbacks.fbsize((GLFWwindow*) window, width, height);
}

// Notifies shared code that a window content scale has changed
// The scale is specified as the ratio between the current and default DPI
//
void _glfwInputWindowContentScale(_GLFWwindow* window, float xscale, float yscale)
{
    if (window->callbacks.scale)
        window->callbacks.scale((GLFWwindow*) window, xscale, yscale);
}

// Notifies shared code that the window contents needs updating
//
void _glfwInputWindowDamage(_GLFWwindow* window)
{
    if (window->callbacks.refresh)
        window->callbacks.refresh((GLFWwindow*) window);
}

// Notifies shared code that the user wishes to close a window
//
void _glfwInputWindowCloseRequest(_GLFWwindow* window)
{
    window->shouldClose = GLFW_TRUE;

    if (window->callbacks.close)
        window->callbacks.close((GLFWwindow*) window);
}

// Notifies shared code that a window has changed its desired monitor
//
void _glfwInputWindowMonitor(_GLFWwindow* window, _GLFWmonitor* monitor)
{
    window->monitor = monitor;
}

//////////////////////////////////////////////////////////////////////////
//////                        GLFW public API                       //////
//////////////////////////////////////////////////////////////////////////

GLFWAPI GLFWwindow* glfwCreateWindow(int width, int height,
                                     const char* title,
                                     GLFWmonitor* monitor,
                                     GLFWwindow* share)
{
    _GLFWfbconfig fbconfig;
    _GLFWctxconfig ctxconfig;
    _GLFWwndconfig wndconfig;
    _GLFWwindow* window;

    assert(title != NULL);
    assert(width >= 0);
    assert(height >= 0);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);

    if (width <= 0 || height <= 0)
    {
        _glfwInputError(GLFW_INVALID_VALUE,
                        "Invalid window size %ix%i",
                        width, height);

        return NULL;
    }

    fbconfig  = _glfw.hints.framebuffer;
    ctxconfig = _glfw.hints.context;
    wndconfig = _glfw.hints.window;

    wndconfig.width   = width;
    wndconfig.height  = height;
    wndconfig.title   = title;
    ctxconfig.share   = (_GLFWwindow*) share;

    if (!_glfwIsValidContextConfig(&ctxconfig))
        return NULL;

    window = calloc(1, sizeof(_GLFWwindow));
    window->next = _glfw.windowListHead;
    _glfw.windowListHead = window;

    window->videoMode.width       = width;
    window->videoMode.height      = height;
    window->videoMode.redBits     = fbconfig.redBits;
    window->videoMode.greenBits   = fbconfig.greenBits;
    window->videoMode.blueBits    = fbconfig.blueBits;
    window->videoMode.refreshRate = _glfw.hints.refreshRate;

    window->monitor     = (_GLFWmonitor*) monitor;
    window->resizable   = wndconfig.resizable;
    window->decorated   = wndconfig.decorated;
    window->autoIconify = wndconfig.autoIconify;
    window->floating    = wndconfig.floating;
    window->focusOnShow = wndconfig.focusOnShow;
    window->cursorMode  = GLFW_CURSOR_NORMAL;

    window->minwidth    = GLFW_DONT_CARE;
    window->minheight   = GLFW_DONT_CARE;
    window->maxwidth    = GLFW_DONT_CARE;
    window->maxheight   = GLFW_DONT_CARE;
    window->numer       = GLFW_DONT_CARE;
    window->denom       = GLFW_DONT_CARE;

    // Open the actual window and create its context
    if (!_glfwPlatformCreateWindow(window, &wndconfig, &ctxconfig, &fbconfig))
    {
        glfwDestroyWindow((GLFWwindow*) window);
        return NULL;
    }

    if (ctxconfig.client != GLFW_NO_API)
    {
        if (!_glfwRefreshContextAttribs(window, &ctxconfig))
        {
            glfwDestroyWindow((GLFWwindow*) window);
            return NULL;
        }
    }

    if (window->monitor)
    {
        if (wndconfig.centerCursor)
            _glfwCenterCursorInContentArea(window);
    }
    else
    {
        if (wndconfig.visible)
        {
            _glfwPlatformShowWindow(window);
            if (wndconfig.focused)
                _glfwPlatformFocusWindow(window);
        }
    }

    return (GLFWwindow*) window;
}

void glfwDefaultWindowHints(void)
{
    _GLFW_REQUIRE_INIT();

    // The default is OpenGL with minimum version 1.0
    memset(&_glfw.hints.context, 0, sizeof(_glfw.hints.context));
    _glfw.hints.context.client = GLFW_OPENGL_API;
    _glfw.hints.context.source = GLFW_NATIVE_CONTEXT_API;
    _glfw.hints.context.major  = 1;
    _glfw.hints.context.minor  = 0;

    // The default is a focused, visible, resizable window with decorations
    memset(&_glfw.hints.window, 0, sizeof(_glfw.hints.window));
    _glfw.hints.window.resizable    = GLFW_TRUE;
    _glfw.hints.window.visible      = GLFW_TRUE;
    _glfw.hints.window.decorated    = GLFW_TRUE;
    _glfw.hints.window.focused      = GLFW_TRUE;
    _glfw.hints.window.autoIconify  = GLFW_TRUE;
    _glfw.hints.window.centerCursor = GLFW_TRUE;
    _glfw.hints.window.focusOnShow  = GLFW_TRUE;

    // The default is 24 bits of color, 24 bits of depth and 8 bits of stencil,
    // double buffered
    memset(&_glfw.hints.framebuffer, 0, sizeof(_glfw.hints.framebuffer));
    _glfw.hints.framebuffer.redBits      = 8;
    _glfw.hints.framebuffer.greenBits    = 8;
    _glfw.hints.framebuffer.blueBits     = 8;
    _glfw.hints.framebuffer.alphaBits    = 8;
    _glfw.hints.framebuffer.depthBits    = 24;
    _glfw.hints.framebuffer.stencilBits  = 8;
    _glfw.hints.framebuffer.doublebuffer = GLFW_TRUE;

    // The default is to select the highest available refresh rate
    _glfw.hints.refreshRate = GLFW_DONT_CARE;

    // The default is to use full Retina resolution framebuffers
    _glfw.hints.window.ns.retina = GLFW_TRUE;
}

GLFWAPI void glfwWindowHint(int hint, int value)
{
    _GLFW_REQUIRE_INIT();

    switch (hint)
    {
        case GLFW_RED_BITS:
            _glfw.hints.framebuffer.redBits = value;
            return;
        case GLFW_GREEN_BITS:
            _glfw.hints.framebuffer.greenBits = value;
            return;
        case GLFW_BLUE_BITS:
            _glfw.hints.framebuffer.blueBits = value;
            return;
        case GLFW_ALPHA_BITS:
            _glfw.hints.framebuffer.alphaBits = value;
            return;
        case GLFW_DEPTH_BITS:
            _glfw.hints.framebuffer.depthBits = value;
            return;
        case GLFW_STENCIL_BITS:
            _glfw.hints.framebuffer.stencilBits = value;
            return;
        case GLFW_ACCUM_RED_BITS:
            _glfw.hints.framebuffer.accumRedBits = value;
            return;
        case GLFW_ACCUM_GREEN_BITS:
            _glfw.hints.framebuffer.accumGreenBits = value;
            return;
        case GLFW_ACCUM_BLUE_BITS:
            _glfw.hints.framebuffer.accumBlueBits = value;
            return;
        case GLFW_ACCUM_ALPHA_BITS:
            _glfw.hints.framebuffer.accumAlphaBits = value;
            return;
        case GLFW_AUX_BUFFERS:
            _glfw.hints.framebuffer.auxBuffers = value;
            return;
        case GLFW_STEREO:
            _glfw.hints.framebuffer.stereo = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_DOUBLEBUFFER:
            _glfw.hints.framebuffer.doublebuffer = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_TRANSPARENT_FRAMEBUFFER:
            _glfw.hints.framebuffer.transparent = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_SAMPLES:
            _glfw.hints.framebuffer.samples = value;
            return;
        case GLFW_SRGB_CAPABLE:
            _glfw.hints.framebuffer.sRGB = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_RESIZABLE:
            _glfw.hints.window.resizable = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_DECORATED:
            _glfw.hints.window.decorated = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_FOCUSED:
            _glfw.hints.window.focused = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_AUTO_ICONIFY:
            _glfw.hints.window.autoIconify = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_FLOATING:
            _glfw.hints.window.floating = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_MAXIMIZED:
            _glfw.hints.window.maximized = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_VISIBLE:
            _glfw.hints.window.visible = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_COCOA_RETINA_FRAMEBUFFER:
            _glfw.hints.window.ns.retina = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_WIN32_KEYBOARD_MENU:
            _glfw.hints.window.win32.keymenu = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_COCOA_GRAPHICS_SWITCHING:
            _glfw.hints.context.nsgl.offline = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_SCALE_TO_MONITOR:
            _glfw.hints.window.scaleToMonitor = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_CENTER_CURSOR:
            _glfw.hints.window.centerCursor = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_FOCUS_ON_SHOW:
            _glfw.hints.window.focusOnShow = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_CLIENT_API:
            _glfw.hints.context.client = value;
            return;
        case GLFW_CONTEXT_CREATION_API:
            _glfw.hints.context.source = value;
            return;
        case GLFW_CONTEXT_VERSION_MAJOR:
            _glfw.hints.context.major = value;
            return;
        case GLFW_CONTEXT_VERSION_MINOR:
            _glfw.hints.context.minor = value;
            return;
        case GLFW_CONTEXT_ROBUSTNESS:
            _glfw.hints.context.robustness = value;
            return;
        case GLFW_OPENGL_FORWARD_COMPAT:
            _glfw.hints.context.forward = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_OPENGL_DEBUG_CONTEXT:
            _glfw.hints.context.debug = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_CONTEXT_NO_ERROR:
            _glfw.hints.context.noerror = value ? GLFW_TRUE : GLFW_FALSE;
            return;
        case GLFW_OPENGL_PROFILE:
            _glfw.hints.context.profile = value;
            return;
        case GLFW_CONTEXT_RELEASE_BEHAVIOR:
            _glfw.hints.context.release = value;
            return;
        case GLFW_REFRESH_RATE:
            _glfw.hints.refreshRate = value;
            return;
    }

    _glfwInputError(GLFW_INVALID_ENUM, "Invalid window hint 0x%08X", hint);
}

GLFWAPI void glfwWindowHintString(int hint, const char* value)
{
    assert(value != NULL);

    _GLFW_REQUIRE_INIT();

    switch (hint)
    {
        case GLFW_COCOA_FRAME_NAME:
            strncpy(_glfw.hints.window.ns.frameName, value,
                    sizeof(_glfw.hints.window.ns.frameName) - 1);
            return;
        case GLFW_X11_CLASS_NAME:
            strncpy(_glfw.hints.window.x11.className, value,
                    sizeof(_glfw.hints.window.x11.className) - 1);
            return;
        case GLFW_X11_INSTANCE_NAME:
            strncpy(_glfw.hints.window.x11.instanceName, value,
                    sizeof(_glfw.hints.window.x11.instanceName) - 1);
            return;
    }

    _glfwInputError(GLFW_INVALID_ENUM, "Invalid window hint string 0x%08X", hint);
}

GLFWAPI void glfwDestroyWindow(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;

    _GLFW_REQUIRE_INIT();

    // Allow closing of NULL (to match the behavior of free)
    if (window == NULL)
        return;

    // Clear all callbacks to avoid exposing a half torn-down window object
    memset(&window->callbacks, 0, sizeof(window->callbacks));

    // The window's context must not be current on another thread when the
    // window is destroyed
    if (window == _glfwPlatformGetTls(&_glfw.contextSlot))
        glfwMakeContextCurrent(NULL);

    _glfwPlatformDestroyWindow(window);

    // Unlink window from global linked list
    {
        _GLFWwindow** prev = &_glfw.windowListHead;

        while (*prev != window)
            prev = &((*prev)->next);

        *prev = window->next;
    }

    free(window);
}

GLFWAPI int glfwWindowShouldClose(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(0);
    return window->shouldClose;
}

GLFWAPI void glfwSetWindowShouldClose(GLFWwindow* handle, int value)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();
    window->shouldClose = value;
}

GLFWAPI void glfwSetWindowTitle(GLFWwindow* handle, const char* title)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);
    assert(title != NULL);

    _GLFW_REQUIRE_INIT();
    _glfwPlatformSetWindowTitle(window, title);
}

GLFWAPI void glfwSetWindowIcon(GLFWwindow* handle,
                               int count, const GLFWimage* images)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);
    assert(count >= 0);
    assert(count == 0 || images != NULL);

    _GLFW_REQUIRE_INIT();
    _glfwPlatformSetWindowIcon(window, count, images);
}

GLFWAPI void glfwGetWindowPos(GLFWwindow* handle, int* xpos, int* ypos)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    if (xpos)
        *xpos = 0;
    if (ypos)
        *ypos = 0;

    _GLFW_REQUIRE_INIT();
    _glfwPlatformGetWindowPos(window, xpos, ypos);
}

GLFWAPI void glfwSetWindowPos(GLFWwindow* handle, int xpos, int ypos)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();

    if (window->monitor)
        return;

    _glfwPlatformSetWindowPos(window, xpos, ypos);
}

GLFWAPI void glfwGetWindowSize(GLFWwindow* handle, int* width, int* height)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    if (width)
        *width = 0;
    if (height)
        *height = 0;

    _GLFW_REQUIRE_INIT();
    _glfwPlatformGetWindowSize(window, width, height);
}

GLFWAPI void glfwSetWindowSize(GLFWwindow* handle, int width, int height)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);
    assert(width >= 0);
    assert(height >= 0);

    _GLFW_REQUIRE_INIT();

    window->videoMode.width  = width;
    window->videoMode.height = height;

    _glfwPlatformSetWindowSize(window, width, height);
}

GLFWAPI void glfwSetWindowSizeLimits(GLFWwindow* handle,
                                     int minwidth, int minheight,
                                     int maxwidth, int maxheight)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();

    if (minwidth != GLFW_DONT_CARE && minheight != GLFW_DONT_CARE)
    {
        if (minwidth < 0 || minheight < 0)
        {
            _glfwInputError(GLFW_INVALID_VALUE,
                            "Invalid window minimum size %ix%i",
                            minwidth, minheight);
            return;
        }
    }

    if (maxwidth != GLFW_DONT_CARE && maxheight != GLFW_DONT_CARE)
    {
        if (maxwidth < 0 || maxheight < 0 ||
            maxwidth < minwidth || maxheight < minheight)
        {
            _glfwInputError(GLFW_INVALID_VALUE,
                            "Invalid window maximum size %ix%i",
                            maxwidth, maxheight);
            return;
        }
    }

    window->minwidth  = minwidth;
    window->minheight = minheight;
    window->maxwidth  = maxwidth;
    window->maxheight = maxheight;

    if (window->monitor || !window->resizable)
        return;

    _glfwPlatformSetWindowSizeLimits(window,
                                     minwidth, minheight,
                                     maxwidth, maxheight);
}

GLFWAPI void glfwSetWindowAspectRatio(GLFWwindow* handle, int numer, int denom)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);
    assert(numer != 0);
    assert(denom != 0);

    _GLFW_REQUIRE_INIT();

    if (numer != GLFW_DONT_CARE && denom != GLFW_DONT_CARE)
    {
        if (numer <= 0 || denom <= 0)
        {
            _glfwInputError(GLFW_INVALID_VALUE,
                            "Invalid window aspect ratio %i:%i",
                            numer, denom);
            return;
        }
    }

    window->numer = numer;
    window->denom = denom;

    if (window->monitor || !window->resizable)
        return;

    _glfwPlatformSetWindowAspectRatio(window, numer, denom);
}

GLFWAPI void glfwGetFramebufferSize(GLFWwindow* handle, int* width, int* height)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    if (width)
        *width = 0;
    if (height)
        *height = 0;

    _GLFW_REQUIRE_INIT();
    _glfwPlatformGetFramebufferSize(window, width, height);
}

GLFWAPI void glfwGetWindowFrameSize(GLFWwindow* handle,
                                    int* left, int* top,
                                    int* right, int* bottom)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    if (left)
        *left = 0;
    if (top)
        *top = 0;
    if (right)
        *right = 0;
    if (bottom)
        *bottom = 0;

    _GLFW_REQUIRE_INIT();
    _glfwPlatformGetWindowFrameSize(window, left, top, right, bottom);
}

GLFWAPI void glfwGetWindowContentScale(GLFWwindow* handle,
                                       float* xscale, float* yscale)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    if (xscale)
        *xscale = 0.f;
    if (yscale)
        *yscale = 0.f;

    _GLFW_REQUIRE_INIT();
    _glfwPlatformGetWindowContentScale(window, xscale, yscale);
}

GLFWAPI float glfwGetWindowOpacity(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(1.f);
    return _glfwPlatformGetWindowOpacity(window);
}

GLFWAPI void glfwSetWindowOpacity(GLFWwindow* handle, float opacity)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);
    assert(opacity == opacity);
    assert(opacity >= 0.f);
    assert(opacity <= 1.f);

    _GLFW_REQUIRE_INIT();

    if (opacity != opacity || opacity < 0.f || opacity > 1.f)
    {
        _glfwInputError(GLFW_INVALID_VALUE, "Invalid window opacity %f", opacity);
        return;
    }

    _glfwPlatformSetWindowOpacity(window, opacity);
}

GLFWAPI void glfwIconifyWindow(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();
    _glfwPlatformIconifyWindow(window);
}

GLFWAPI void glfwRestoreWindow(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();
    _glfwPlatformRestoreWindow(window);
}

GLFWAPI void glfwMaximizeWindow(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();

    if (window->monitor)
        return;

    _glfwPlatformMaximizeWindow(window);
}

GLFWAPI void glfwShowWindow(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();

    if (window->monitor)
        return;

    _glfwPlatformShowWindow(window);

    if (window->focusOnShow)
        _glfwPlatformFocusWindow(window);
}

GLFWAPI void glfwRequestWindowAttention(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();

    _glfwPlatformRequestWindowAttention(window);
}

GLFWAPI void glfwHideWindow(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();

    if (window->monitor)
        return;

    _glfwPlatformHideWindow(window);
}

GLFWAPI void glfwFocusWindow(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();

    _glfwPlatformFocusWindow(window);
}

GLFWAPI int glfwGetWindowAttrib(GLFWwindow* handle, int attrib)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(0);

    switch (attrib)
    {
        case GLFW_FOCUSED:
            return _glfwPlatformWindowFocused(window);
        case GLFW_ICONIFIED:
            return _glfwPlatformWindowIconified(window);
        case GLFW_VISIBLE:
            return _glfwPlatformWindowVisible(window);
        case GLFW_MAXIMIZED:
            return _glfwPlatformWindowMaximized(window);
        case GLFW_HOVERED:
            return _glfwPlatformWindowHovered(window);
        case GLFW_FOCUS_ON_SHOW:
            return window->focusOnShow;
        case GLFW_TRANSPARENT_FRAMEBUFFER:
            return _glfwPlatformFramebufferTransparent(window);
        case GLFW_RESIZABLE:
            return window->resizable;
        case GLFW_DECORATED:
            return window->decorated;
        case GLFW_FLOATING:
            return window->floating;
        case GLFW_AUTO_ICONIFY:
            return window->autoIconify;
        case GLFW_CLIENT_API:
            return window->context.client;
        case GLFW_CONTEXT_CREATION_API:
            return window->context.source;
        case GLFW_CONTEXT_VERSION_MAJOR:
            return window->context.major;
        case GLFW_CONTEXT_VERSION_MINOR:
            return window->context.minor;
        case GLFW_CONTEXT_REVISION:
            return window->context.revision;
        case GLFW_CONTEXT_ROBUSTNESS:
            return window->context.robustness;
        case GLFW_OPENGL_FORWARD_COMPAT:
            return window->context.forward;
        case GLFW_OPENGL_DEBUG_CONTEXT:
            return window->context.debug;
        case GLFW_OPENGL_PROFILE:
            return window->context.profile;
        case GLFW_CONTEXT_RELEASE_BEHAVIOR:
            return window->context.release;
        case GLFW_CONTEXT_NO_ERROR:
            return window->context.noerror;
    }

    _glfwInputError(GLFW_INVALID_ENUM, "Invalid window attribute 0x%08X", attrib);
    return 0;
}

GLFWAPI void glfwSetWindowAttrib(GLFWwindow* handle, int attrib, int value)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();

    value = value ? GLFW_TRUE : GLFW_FALSE;

    if (attrib == GLFW_AUTO_ICONIFY)
        window->autoIconify = value;
    else if (attrib == GLFW_RESIZABLE)
    {
        if (window->resizable == value)
            return;

        window->resizable = value;
        if (!window->monitor)
            _glfwPlatformSetWindowResizable(window, value);
    }
    else if (attrib == GLFW_DECORATED)
    {
        if (window->decorated == value)
            return;

        window->decorated = value;
        if (!window->monitor)
            _glfwPlatformSetWindowDecorated(window, value);
    }
    else if (attrib == GLFW_FLOATING)
    {
        if (window->floating == value)
            return;

        window->floating = value;
        if (!window->monitor)
            _glfwPlatformSetWindowFloating(window, value);
    }
    else if (attrib == GLFW_FOCUS_ON_SHOW)
        window->focusOnShow = value;
    else
        _glfwInputError(GLFW_INVALID_ENUM, "Invalid window attribute 0x%08X", attrib);
}

GLFWAPI GLFWmonitor* glfwGetWindowMonitor(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    return (GLFWmonitor*) window->monitor;
}

GLFWAPI void glfwSetWindowMonitor(GLFWwindow* wh,
                                  GLFWmonitor* mh,
                                  int xpos, int ypos,
                                  int width, int height,
                                  int refreshRate)
{
    _GLFWwindow* window = (_GLFWwindow*) wh;
    _GLFWmonitor* monitor = (_GLFWmonitor*) mh;
    assert(window != NULL);
    assert(width >= 0);
    assert(height >= 0);

    _GLFW_REQUIRE_INIT();

    if (width <= 0 || height <= 0)
    {
        _glfwInputError(GLFW_INVALID_VALUE,
                        "Invalid window size %ix%i",
                        width, height);
        return;
    }

    if (refreshRate < 0 && refreshRate != GLFW_DONT_CARE)
    {
        _glfwInputError(GLFW_INVALID_VALUE,
                        "Invalid refresh rate %i",
                        refreshRate);
        return;
    }

    window->videoMode.width       = width;
    window->videoMode.height      = height;
    window->videoMode.refreshRate = refreshRate;

    _glfwPlatformSetWindowMonitor(window, monitor,
                                  xpos, ypos, width, height,
                                  refreshRate);
}

GLFWAPI void glfwSetWindowUserPointer(GLFWwindow* handle, void* pointer)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT();
    window->userPointer = pointer;
}

GLFWAPI void* glfwGetWindowUserPointer(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    return window->userPointer;
}

GLFWAPI GLFWwindowposfun glfwSetWindowPosCallback(GLFWwindow* handle,
                                                  GLFWwindowposfun cbfun)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    _GLFW_SWAP_POINTERS(window->callbacks.pos, cbfun);
    return cbfun;
}

GLFWAPI GLFWwindowsizefun glfwSetWindowSizeCallback(GLFWwindow* handle,
                                                    GLFWwindowsizefun cbfun)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    _GLFW_SWAP_POINTERS(window->callbacks.size, cbfun);
    return cbfun;
}

GLFWAPI GLFWwindowclosefun glfwSetWindowCloseCallback(GLFWwindow* handle,
                                                      GLFWwindowclosefun cbfun)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    _GLFW_SWAP_POINTERS(window->callbacks.close, cbfun);
    return cbfun;
}

GLFWAPI GLFWwindowrefreshfun glfwSetWindowRefreshCallback(GLFWwindow* handle,
                                                          GLFWwindowrefreshfun cbfun)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    _GLFW_SWAP_POINTERS(window->callbacks.refresh, cbfun);
    return cbfun;
}

GLFWAPI GLFWwindowfocusfun glfwSetWindowFocusCallback(GLFWwindow* handle,
                                                      GLFWwindowfocusfun cbfun)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    _GLFW_SWAP_POINTERS(window->callbacks.focus, cbfun);
    return cbfun;
}

GLFWAPI GLFWwindowiconifyfun glfwSetWindowIconifyCallback(GLFWwindow* handle,
                                                          GLFWwindowiconifyfun cbfun)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    _GLFW_SWAP_POINTERS(window->callbacks.iconify, cbfun);
    return cbfun;
}

GLFWAPI GLFWwindowmaximizefun glfwSetWindowMaximizeCallback(GLFWwindow* handle,
                                                            GLFWwindowmaximizefun cbfun)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    _GLFW_SWAP_POINTERS(window->callbacks.maximize, cbfun);
    return cbfun;
}

GLFWAPI GLFWframebuffersizefun glfwSetFramebufferSizeCallback(GLFWwindow* handle,
                                                              GLFWframebuffersizefun cbfun)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    _GLFW_SWAP_POINTERS(window->callbacks.fbsize, cbfun);
    return cbfun;
}

GLFWAPI GLFWwindowcontentscalefun glfwSetWindowContentScaleCallback(GLFWwindow* handle,
                                                                    GLFWwindowcontentscalefun cbfun)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    assert(window != NULL);

    _GLFW_REQUIRE_INIT_OR_RETURN(NULL);
    _GLFW_SWAP_POINTERS(window->callbacks.scale, cbfun);
    return cbfun;
}

GLFWAPI void glfwPollEvents(void)
{
    _GLFW_REQUIRE_INIT();
    _glfwPlatformPollEvents();
}

GLFWAPI void glfwWaitEvents(void)
{
    _GLFW_REQUIRE_INIT();
    _glfwPlatformWaitEvents();
}

GLFWAPI void glfwWaitEventsTimeout(double timeout)
{
    _GLFW_REQUIRE_INIT();
    assert(timeout == timeout);
    assert(timeout >= 0.0);
    assert(timeout <= DBL_MAX);

    if (timeout != timeout || timeout < 0.0 || timeout > DBL_MAX)
    {
        _glfwInputError(GLFW_INVALID_VALUE, "Invalid time %f", timeout);
        return;
    }

    _glfwPlatformWaitEventsTimeout(timeout);
}

GLFWAPI void glfwPostEmptyEvent(void)
{
    _GLFW_REQUIRE_INIT();
    _glfwPlatformPostEmptyEvent();
}
