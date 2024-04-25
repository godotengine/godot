//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GLCONTEXT_H
#define MATERIALX_GLCONTEXT_H

/// @file
/// OpenGL context class

#include <MaterialXRenderGlsl/Export.h>

#include <MaterialXRenderHw/SimpleWindow.h>

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#elif defined(__linux__) || defined(__FreeBSD__)
#include <GL/glx.h>
#endif

MATERIALX_NAMESPACE_BEGIN

/// Platform dependent definition of a hardware context
#if defined(_WIN32)
using HardwareContextHandle = HGLRC;
#elif defined(__linux__) || defined(__FreeBSD__)
using HardwareContextHandle = GLXContext;
#else
using HardwareContextHandle = void*;
#endif

/// SimpleWindow shared pointer
using SimpleWindowPtr = std::shared_ptr<class SimpleWindow>;

/// GLContext shared pointer
using GLContextPtr = std::shared_ptr<class GLContext>;

/// @class GLContext
/// An OpenGL context singleton
class MX_RENDERGLSL_API GLContext
{
  public:
    /// Create a new context
    static GLContextPtr create(SimpleWindowPtr window, HardwareContextHandle context = {})
    {
        return GLContextPtr(new GLContext(window, context));
    }

    /// Default destructor
    virtual ~GLContext();

    /// Return OpenGL context handle
    HardwareContextHandle contextHandle() const
    {
        return _contextHandle;
    }

    /// Return if context is valid
    bool isValid() const
    {
        return _isValid;
    }

    /// Make the context "current" before execution of OpenGL operations
    int makeCurrent();

  protected:
    // Create the base context. A OpenGL context to share with can be passed in.
    GLContext(SimpleWindowPtr window, HardwareContextHandle context = 0);

    // Simple window
    SimpleWindowPtr _window;

    // Context handle
    HardwareContextHandle _contextHandle;

    // Flag to indicate validity
    bool _isValid;

#if defined(__linux__) || defined(__FreeBSD__)
    // An X window used by context operations
    Window _xWindow;

    // An X display used by context operations
    Display* _xDisplay;
#endif
};

MATERIALX_NAMESPACE_END

#endif
