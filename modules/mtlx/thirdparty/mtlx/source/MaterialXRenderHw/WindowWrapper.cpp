//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRenderHw/WindowWrapper.h>

#if defined(__linux__) || defined(__FreeBSD__)
#include <X11/Intrinsic.h>
#elif defined(__APPLE__)
#include <MaterialXRenderHw/WindowCocoaWrappers.h>
#endif

MATERIALX_NAMESPACE_BEGIN

#if defined(_WIN32)

WindowWrapper::WindowWrapper(ExternalWindowHandle externalHandle,
                             InternalWindowHandle internalHandle,
                             DisplayHandle /*display*/)
{
    _externalHandle = externalHandle;
    if (_externalHandle && !internalHandle)
    {
        // Cache a HDC that corresponds to the window handle.
        _internalHandle = GetDC(_externalHandle);
    }
    else
    {
        _internalHandle = internalHandle;
    }
}

WindowWrapper::~WindowWrapper()
{
    release();
}

void WindowWrapper::release()
{
    if (_externalHandle)
    {
        // Release acquired DC
        ReleaseDC(_externalHandle, _internalHandle);
    }
    _externalHandle = 0;
    _internalHandle = 0;
}

#elif defined(__linux__) || defined(__FreeBSD__)

WindowWrapper::WindowWrapper(ExternalWindowHandle externalHandle,
                             InternalWindowHandle internalHandle,
                             DisplayHandle display)
{
    _xDisplay = display;
    _framebufferWindow = 0;
    _externalHandle = externalHandle;
    // Cache a pointer to the window.
    if (internalHandle)
        _internalHandle = internalHandle;
    else
        _internalHandle = XtWindow(externalHandle);
}

WindowWrapper::~WindowWrapper()
{
    release();
}

void WindowWrapper::release()
{
    // No explicit release calls are required.
    _externalHandle = 0;
    _internalHandle = 0;
    _framebufferWindow = 0;
    _xDisplay = 0;
}

#elif defined(__APPLE__)

WindowWrapper::WindowWrapper(ExternalWindowHandle externalHandle,
                             InternalWindowHandle internalHandle,
                             DisplayHandle display)
{
    _externalHandle = externalHandle;
#ifndef TARGET_OS_IOS
    // Cache a pointer to the window.
    _internalHandle = NSUtilGetView(externalHandle);
#else
    _internalHandle = nullptr;
#endif
}

WindowWrapper::~WindowWrapper()
{
    release();
}

void WindowWrapper::release()
{
    // No explicit release calls are required.
    _externalHandle = 0;
    _internalHandle = 0;
}

#endif

//
// Creator
//

WindowWrapperPtr WindowWrapper::create(ExternalWindowHandle externalHandle,
                                       InternalWindowHandle internalHandle,
                                       DisplayHandle display)
{
    return std::shared_ptr<WindowWrapper>(new WindowWrapper(externalHandle, internalHandle, display));
}

MATERIALX_NAMESPACE_END
