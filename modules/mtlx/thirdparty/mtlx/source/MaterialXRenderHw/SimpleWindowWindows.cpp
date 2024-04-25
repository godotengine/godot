//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_WIN32)

#include <MaterialXRenderHw/SimpleWindow.h>
#include <MaterialXRenderHw/WindowWrapper.h>
#include <string>

MATERIALX_NAMESPACE_BEGIN

SimpleWindow::SimpleWindow() :
    _width(0),
    _height(0)
{
    // Give a unique ID to this window.
    //
    static unsigned int windowCount = 1;
    _id = windowCount;
    windowCount++;

    // Generate a unique string for our window class.
    sprintf_s(_windowClassName, "_SW_%u", _id);
}

// No-op window procedure
LRESULT CALLBACK NoOpProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
        case WM_CLOSE:
        case WM_DESTROY:
            break;
        default:
            return DefWindowProc(hWnd, msg, wParam, lParam);
            break;
    }
    return 0;
}

bool SimpleWindow::initialize(const char* title,
                              unsigned int width, unsigned int height,
                              void* /*applicationShell*/)
{
    HINSTANCE hInstance = GetModuleHandle(NULL);

    // Basic windows class structure
    //
    WNDCLASS wc;
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wc.lpfnWndProc = (WNDPROC) NoOpProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance; // Set the instance to this application
    wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = NULL; // No background required
    wc.lpszMenuName = NULL;  // No menu required
    wc.lpszClassName = _windowClassName;

    if (!RegisterClass(&wc))
    {
        _id = 0;
        return false;
    }

    // Window style and extended style
    //
    DWORD dwStyle = WS_OVERLAPPEDWINDOW;
    DWORD dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;

    // Set the rectangle of the client area.
    RECT WindowRect;
    WindowRect.left = (long) 0;
    WindowRect.top = (long) 0;
    WindowRect.right = (long) width;
    WindowRect.bottom = (long) height;

    // Calculate the exact window size (including border) so that the
    // client area has the desired dimensions.
    //
    AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle);

    // Attempt to create the window.
    HWND hWnd = CreateWindowEx(dwExStyle, _windowClassName, title,
        dwStyle | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
        0, 0,                               // Window position
        WindowRect.right - WindowRect.left, // Window width (including borders)
        WindowRect.bottom - WindowRect.top, // Window height (including borders/title bar)
        NULL,                               // No parent window
        NULL,                               // No menu
        hInstance,                          // Instance
        NULL);                              // Don't pass anything To WM_CREATE

    if (!hWnd)
    {
        _id = 0;
        return false;
    }

    _windowWrapper = WindowWrapper::create(hWnd);

    return true;
}

SimpleWindow::~SimpleWindow()
{
    HWND hWnd = _windowWrapper->externalHandle();
    if (hWnd)
    {
        _windowWrapper->release();
    }

    DestroyWindow(hWnd);
    UnregisterClass(_windowClassName, GetModuleHandle(NULL));
}

MATERIALX_NAMESPACE_END

#endif
