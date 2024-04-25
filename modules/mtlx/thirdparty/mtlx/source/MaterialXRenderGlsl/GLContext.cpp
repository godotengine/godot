//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__) || defined(__FreeBSD__)
#include <X11/Intrinsic.h>
#elif defined(__APPLE__)
#include <MaterialXRenderHw/WindowCocoaWrappers.h>
#include <MaterialXRenderGlsl/GLCocoaWrappers.h>
#endif

#include <MaterialXRenderGlsl/External/Glad/glad.h>
#include <MaterialXRenderGlsl/GLContext.h>

MATERIALX_NAMESPACE_BEGIN

#if defined(_WIN32)

GLContext::GLContext(SimpleWindowPtr window, HardwareContextHandle sharedWithContext) :
    _window(window),
    _contextHandle(nullptr),
    _isValid(false)
{
    // Get the existing window wrapper.
    WindowWrapperPtr windowWrapper = _window->getWindowWrapper();
    if (!windowWrapper->isValid())
    {
        return;
    }

    // Use a generic pixel format to create the context
    static PIXELFORMATDESCRIPTOR pfd =
    {
        sizeof(PIXELFORMATDESCRIPTOR),
        1,
        PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        PFD_TYPE_RGBA,
        32,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        16, 0, 0, PFD_MAIN_PLANE, 0, 0, 0, 0
    };

    int chosenPixelFormat = ChoosePixelFormat(windowWrapper->internalHandle(), &pfd);
    if (chosenPixelFormat)
    {
        if (SetPixelFormat(windowWrapper->internalHandle(), chosenPixelFormat, &pfd))
        {
            _contextHandle = wglCreateContext(windowWrapper->internalHandle());
            if (_contextHandle)
            {
                if (sharedWithContext)
                {
                    wglShareLists(_contextHandle, sharedWithContext);
                }

                int makeCurrentOk = wglMakeCurrent(windowWrapper->internalHandle(), _contextHandle);
                if (makeCurrentOk)
                {
                    _isValid = true;
                }
            }
        }
    }
}

#elif defined(__linux__) || defined(__FreeBSD__)

GLContext::GLContext(const SimpleWindowPtr window, HardwareContextHandle sharedWithContext) :
    _window(window),
    _contextHandle(nullptr),
    _isValid(false)
{
    // Get the existing window wrapper and X display.
    WindowWrapperPtr windowWrapper = _window->getWindowWrapper();
    if (!windowWrapper->isValid())
    {
        return;
    }
    _xDisplay = windowWrapper->getXDisplay();

    // Find an appropriate OpenGL-capable visual.
    static int attr[] = { GLX_RGBA,
                          GLX_DOUBLEBUFFER,
                          GLX_RED_SIZE, 8,
                          GLX_GREEN_SIZE, 8,
                          GLX_BLUE_SIZE, 8,
                          GLX_DEPTH_SIZE, 24,
                          GLX_STENCIL_SIZE, 8,
                          None };
    XVisualInfo* vi = glXChooseVisual(_xDisplay, DefaultScreen(_xDisplay), attr);
    if (!vi)
    {
        return;
    }

    // Create an OpenGL rendering context.
    _contextHandle = glXCreateContext(_xDisplay, vi, sharedWithContext, GL_TRUE);
    if (!_contextHandle)
    {
        return;
    }

    // Create an X colormap and window.
    Colormap cmap = XCreateColormap(_xDisplay, RootWindow(_xDisplay, vi->screen), vi->visual, AllocNone);
    XSetWindowAttributes swa;
    swa.colormap = cmap;
    swa.border_pixel = 0;
    swa.background_pixmap = None;
    _xWindow = XCreateWindow(_xDisplay, RootWindow(_xDisplay, vi->screen),
                             0, 0, 10, 10, 0, vi->depth, InputOutput, vi->visual,
                             CWBackPixmap | CWBorderPixel | CWColormap, &swa);
    if (!_xWindow)
    {
        return;
    }

    glXMakeCurrent(_xDisplay, _xWindow, _contextHandle);
    _isValid = true;
}

#elif defined(__APPLE__)

GLContext::GLContext(const SimpleWindowPtr window, HardwareContextHandle sharedWithContext) :
    _window(window),
    _contextHandle(nullptr),
    _isValid(false)
{
    void* pixelFormat = NSOpenGLChoosePixelFormatWrapper(true, 0, 32, 24, 8, 0, 0, false,
        false, false, false, false);
    if (!pixelFormat)
    {
        return;
    }

    // Create the context, but do not share against other contexts.
    // (Instead, all other contexts will share against this one.)
    _contextHandle = NSOpenGLCreateContextWrapper(pixelFormat, sharedWithContext);
    NSOpenGLReleasePixelFormat(pixelFormat);
    NSOpenGLMakeCurrent(_contextHandle);

    _isValid = true;
}

#endif

GLContext::~GLContext()
{
    if (_isValid)
    {
#if defined(_WIN32)

        wglDeleteContext(_contextHandle);

#elif defined(__linux__) || defined(__FreeBSD__)

        glXMakeCurrent(_xDisplay, None, NULL);

        if (_contextHandle != 0)
        {
            glXDestroyContext(_xDisplay, _contextHandle);
        }
        if (_xWindow != 0)
        {
            XDestroyWindow(_xDisplay, _xWindow);
        }

#elif defined(__APPLE__)

        if (_contextHandle != 0)
        {
            NSOpenGLDestroyCurrentContext(&_contextHandle);
        }

#endif
    }
}

int GLContext::makeCurrent()
{
    if (!_isValid)
    {
        return 0;
    }

    int makeCurrentOk = 0;

#if defined(_WIN32)
    makeCurrentOk = wglMakeCurrent(_window->getWindowWrapper()->internalHandle(), _contextHandle);
#elif defined(__linux__) || defined(__FreeBSD__)
    makeCurrentOk = glXMakeCurrent(_xDisplay, _xWindow, _contextHandle);
#elif defined(__APPLE__)
    NSOpenGLMakeCurrent(_contextHandle);
    if (NSOpenGLGetCurrentContextWrapper() == _contextHandle)
    {
        makeCurrentOk = 1;
    }
#endif

    return makeCurrentOk;
}

MATERIALX_NAMESPACE_END
