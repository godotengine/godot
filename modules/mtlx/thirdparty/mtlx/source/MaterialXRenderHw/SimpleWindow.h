//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SIMPLEWINDOW_H
#define MATERIALX_SIMPLEWINDOW_H

#include <MaterialXCore/Library.h>

#include <MaterialXRenderHw/Export.h>
#include <MaterialXRenderHw/WindowWrapper.h>

MATERIALX_NAMESPACE_BEGIN

/// SimpleWindow shared pointer
using SimpleWindowPtr = std::shared_ptr<class SimpleWindow>;

/// @class SimpleWindow
/// A platform-independent window class.
///
/// Plaform-specific resources are encapsulated by a WindowWrapper instance.
class MX_RENDERHW_API SimpleWindow
{
  public:
    /// Static instance create function
    static SimpleWindowPtr create() { return SimpleWindowPtr(new SimpleWindow); }

    /// Default destructor
    virtual ~SimpleWindow();

    /// Window initialization
    bool initialize(const char* title, unsigned int width, unsigned int height, void* applicationShell);

    /// Return our platform-specific resource wrapper
    WindowWrapperPtr getWindowWrapper()
    {
        return _windowWrapper;
    }

    /// Return width of window
    unsigned int width() const
    {
        return _width;
    }

    /// Return height of window
    unsigned int height() const
    {
        return _height;
    }

    /// Check for validity
    bool isValid() const
    {
        return _windowWrapper && _windowWrapper->isValid();
    }

  protected:
    // Default constructor
    SimpleWindow();

    // Wrapper for platform specific window resources
    WindowWrapperPtr _windowWrapper;

    // Window dimensions
    unsigned int _width;
    unsigned int _height;

    // Unique window identifier generated dynamically at creation time.
    unsigned int _id;

#if defined(_WIN32)
    // Window class name for window generated at creation time.
    char _windowClassName[128];
#endif
};

MATERIALX_NAMESPACE_END

#endif
