//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__linux__) || defined(__FreeBSD__)

#include <MaterialXRenderHw/SimpleWindow.h>

#include <X11/StringDefs.h>
#include <X11/Shell.h> // for applicationShellWidgetClass
#include <X11/Xlib.h> // for XEvent definition
#include <X11/Intrinsic.h> // for XtCallbackProc definition

MATERIALX_NAMESPACE_BEGIN

SimpleWindow::SimpleWindow() :
    _width(0),
    _height(0)
{
    // Give a unique ID to this window.
    static unsigned int windowCount = 1;
    _id = windowCount;
    windowCount++;
}

bool SimpleWindow::initialize(const char* title,
                              unsigned int width, unsigned int height,
                              void* applicationShell)
{
    int n = 0;

    XtAppContext appContext;
    Widget shell;
    static Widget batchShell;
    if (!applicationShell)
    {
        static bool initializedXServer = false;
        // Connect to the X Server
        if (!initializedXServer)
        {
            batchShell = XtOpenApplication(&appContext, "__mx_dummy__app__",
                0, 0, &n, 0, 0,
                applicationShellWidgetClass, 0, 0);
            initializedXServer = true;
        }
        shell = batchShell;
    }
    else
    {
        // Reuse existing application shell;
        shell = (Widget) applicationShell;
    }

    if (!shell)
    {
        _id = 0;
        return false;
    }

    Arg args[6];
    n = 0;
    XtSetArg(args[n], XtNx, 0); n++;
    XtSetArg(args[n], XtNy, 0); n++;
    XtSetArg(args[n], XtNwidth, width); n++;
    XtSetArg(args[n], XtNheight, height); n++;
    Widget widget = XtCreatePopupShell(title, topLevelShellWidgetClass, shell, args, n);
    if (!widget)
    {
        _id = 0;
        return false;
    }

    XtRealizeWidget(widget);
    _windowWrapper = WindowWrapper::create(widget, XtWindow(widget), XtDisplay(widget));

    return true;
}

SimpleWindow::~SimpleWindow()
{
    if (!_windowWrapper)
    {
        return;
    }

    Widget widget = _windowWrapper->externalHandle();
    if (widget)
    {
        // Unrealize the widget first to avoid X calls to it
        XtUnrealizeWidget(widget);
        XtDestroyWidget(widget);
        widget = nullptr;
    }
}

MATERIALX_NAMESPACE_END

#endif
