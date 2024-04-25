//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__APPLE__)

#ifdef TARGET_OS_IOS

#include <MaterialXRenderHw/SimpleWindow.h>

MATERIALX_NAMESPACE_BEGIN

SimpleWindow::SimpleWindow() :
    _width(0),
    _height(0)
{
    // Give a unique identifier to this window.
    static unsigned int windowCount = 1;
    _id = windowCount;
    windowCount++;
}

bool SimpleWindow::initialize(const char* title,
                              unsigned int width, unsigned int height,
                              void* /*applicationShell*/)
{
    _windowWrapper = WindowWrapper::create(nullptr);
    return true;
}

SimpleWindow::~SimpleWindow()
{
}

MATERIALX_NAMESPACE_END

#endif

#endif
