//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_WINDOWWRAPPER_H
#define MATERIALX_WINDOWWRAPPER_H

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#undef APIENTRY
#include <windows.h>
#elif defined(__linux__) || defined(__FreeBSD__)
#include <X11/X.h> // for Window
#include <X11/Xlib.h> // for Display
using Widget = struct _WidgetRec*;
#endif

#include <memory>
#include <MaterialXCore/Library.h>

MATERIALX_NAMESPACE_BEGIN
/// OS specific type windowing definitions
#if defined(_WIN32)
/// External handle is a window handle
using ExternalWindowHandle = HWND;
/// Internal handle is a device context
using InternalWindowHandle = HDC;
/// Display handle concept has no equivalence on Windows
using DisplayHandle = void*;
#elif defined(__linux__) || defined(__FreeBSD__)
/// External handle is a widget
using ExternalWindowHandle = Widget;
/// Internal handle is the window for the widget
using InternalWindowHandle = Window;
/// Display handle is the X display
using DisplayHandle = Display*;
/// Application shell
using Widget = struct _WidgetRec*;
#elif defined(__APPLE__)
/// External handle is a window handle
using ExternalWindowHandle = void*;
/// Internal handle concept has no equivalence on Mac
using InternalWindowHandle = void*;
/// Display handle concept has no equivalence on Mac
using DisplayHandle = void*;
#else
using Widget = void*;
using ExternalWindowHandle = void*;
using InternalWindowHandle = void*;
using DisplayHandle = void*;
#endif

/// WindowWrapper shared pointer
using WindowWrapperPtr = std::shared_ptr<class WindowWrapper>;

/// @class WindowWrapper
/// Generic wrapper for encapsulating a "window" construct
///
/// Each supported platform will have specific storage and management logic.
class WindowWrapper
{
  public:
    /// Create a new WindowWrapper
    static WindowWrapperPtr create(ExternalWindowHandle externalHandle = {},
                                   InternalWindowHandle internalHandle = {},
                                   DisplayHandle display = {});

    // Default destructor
    virtual ~WindowWrapper();

    /// Return "external" handle
    ExternalWindowHandle externalHandle() const
    {
        return _externalHandle;
    }

    /// Return "internal" handle
    InternalWindowHandle internalHandle() const
    {
        return _internalHandle;
    }

    /// Check that there is a valid OS handle set.
    /// It is sufficient to just check the internal handle.
    bool isValid() const
    {
        return _internalHandle != 0;
    }

    /// Release resources stored in wrapper
    void release();

#if defined(__linux__) || defined(__FreeBSD__)
    /// Return X display
    Display* getXDisplay() const
    {
        return _xDisplay;
    }
#endif

  protected:
    WindowWrapper(ExternalWindowHandle externalHandle,
                  InternalWindowHandle internalHandle,
                  DisplayHandle display);

  protected:
    ExternalWindowHandle _externalHandle;
    InternalWindowHandle _internalHandle;

#if defined(__linux__) || defined(__FreeBSD__)
    /// Window ID of framebuffer instance created in the wrapper
    Window _framebufferWindow;
    /// X Display
    Display* _xDisplay;
#endif
};

MATERIALX_NAMESPACE_END

#endif
