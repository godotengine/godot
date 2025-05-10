/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#ifdef SDL_VIDEO_DRIVER_X11

#include <unistd.h> // For getpid() and readlink()

#include "../../core/linux/SDL_system_theme.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_mouse_c.h"
#include "../SDL_pixels_c.h"
#include "../SDL_sysvideo.h"

#include "SDL_x11framebuffer.h"
#include "SDL_x11pen.h"
#include "SDL_x11touch.h"
#include "SDL_x11video.h"
#include "SDL_x11xfixes.h"
#include "SDL_x11xinput2.h"
#include "SDL_x11messagebox.h"
#include "SDL_x11shape.h"
#include "SDL_x11xsync.h"
#include "SDL_x11xtest.h"

#ifdef SDL_VIDEO_OPENGL_EGL
#include "SDL_x11opengles.h"
#endif

// Initialization/Query functions
static bool X11_VideoInit(SDL_VideoDevice *_this);
static void X11_VideoQuit(SDL_VideoDevice *_this);

// X11 driver bootstrap functions

static void X11_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_VideoData *data = device->internal;
    if (device->vulkan_config.loader_handle) {
        device->Vulkan_UnloadLibrary(device);
    }
    if (data->display) {
        X11_XCloseDisplay(data->display);
    }
    if (data->request_display) {
        X11_XCloseDisplay(data->request_display);
    }
    SDL_free(data->windowlist);
    if (device->wakeup_lock) {
        SDL_DestroyMutex(device->wakeup_lock);
    }
    SDL_free(device->internal);
    SDL_free(device);

    SDL_X11_UnloadSymbols();
}

static bool X11_IsXWayland(Display *d)
{
    int opcode, event, error;
    return X11_XQueryExtension(d, "XWAYLAND", &opcode, &event, &error) == True;
}

static bool X11_CheckCurrentDesktop(const char *name)
{
    SDL_Environment *env = SDL_GetEnvironment();

    const char *desktopVar = SDL_GetEnvironmentVariable(env, "DESKTOP_SESSION");
    if (desktopVar && SDL_strcasecmp(desktopVar, name) == 0) {
        return true;
    }

    desktopVar = SDL_GetEnvironmentVariable(env, "XDG_CURRENT_DESKTOP");
    if (desktopVar && SDL_strcasestr(desktopVar, name)) {
        return true;
    }

    return false;
}

static SDL_VideoDevice *X11_CreateDevice(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *data;
    const char *display = NULL; // Use the DISPLAY environment variable
    Display *x11_display = NULL;

    if (!SDL_X11_LoadSymbols()) {
        return NULL;
    }

    /* Need for threading gl calls. This is also required for the proprietary
        nVidia driver to be threaded. */
    X11_XInitThreads();

    // Open the display first to be sure that X11 is available
    x11_display = X11_XOpenDisplay(display);

    if (!x11_display) {
        SDL_X11_UnloadSymbols();
        return NULL;
    }

    // Initialize all variables that we clean on shutdown
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        return NULL;
    }
    data = (struct SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    if (!data) {
        SDL_free(device);
        return NULL;
    }
    device->internal = data;

    data->global_mouse_changed = true;

#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
    data->active_cursor_confined_window = NULL;
#endif // SDL_VIDEO_DRIVER_X11_XFIXES

    data->display = x11_display;
    data->request_display = X11_XOpenDisplay(display);
    if (!data->request_display) {
        X11_XCloseDisplay(data->display);
        SDL_free(device->internal);
        SDL_free(device);
        SDL_X11_UnloadSymbols();
        return NULL;
    }

    device->wakeup_lock = SDL_CreateMutex();

#ifdef X11_DEBUG
    X11_XSynchronize(data->display, True);
#endif

    /* Steam Deck will have an on-screen keyboard, so check their environment
     * variable so we can make use of SDL_StartTextInput.
     */
    data->is_steam_deck = SDL_GetHintBoolean("SteamDeck", false);

    // Set the function pointers
    device->VideoInit = X11_VideoInit;
    device->VideoQuit = X11_VideoQuit;
    device->ResetTouch = X11_ResetTouch;
    device->GetDisplayModes = X11_GetDisplayModes;
    device->GetDisplayBounds = X11_GetDisplayBounds;
    device->GetDisplayUsableBounds = X11_GetDisplayUsableBounds;
    device->GetWindowICCProfile = X11_GetWindowICCProfile;
    device->SetDisplayMode = X11_SetDisplayMode;
    device->SuspendScreenSaver = X11_SuspendScreenSaver;
    device->PumpEvents = X11_PumpEvents;
    device->WaitEventTimeout = X11_WaitEventTimeout;
    device->SendWakeupEvent = X11_SendWakeupEvent;

    device->CreateSDLWindow = X11_CreateWindow;
    device->SetWindowTitle = X11_SetWindowTitle;
    device->SetWindowIcon = X11_SetWindowIcon;
    device->SetWindowPosition = X11_SetWindowPosition;
    device->SetWindowSize = X11_SetWindowSize;
    device->SetWindowMinimumSize = X11_SetWindowMinimumSize;
    device->SetWindowMaximumSize = X11_SetWindowMaximumSize;
    device->SetWindowAspectRatio = X11_SetWindowAspectRatio;
    device->GetWindowBordersSize = X11_GetWindowBordersSize;
    device->SetWindowOpacity = X11_SetWindowOpacity;
    device->SetWindowParent = X11_SetWindowParent;
    device->SetWindowModal = X11_SetWindowModal;
    device->ShowWindow = X11_ShowWindow;
    device->HideWindow = X11_HideWindow;
    device->RaiseWindow = X11_RaiseWindow;
    device->MaximizeWindow = X11_MaximizeWindow;
    device->MinimizeWindow = X11_MinimizeWindow;
    device->RestoreWindow = X11_RestoreWindow;
    device->SetWindowBordered = X11_SetWindowBordered;
    device->SetWindowResizable = X11_SetWindowResizable;
    device->SetWindowAlwaysOnTop = X11_SetWindowAlwaysOnTop;
    device->SetWindowFullscreen = X11_SetWindowFullscreen;
    device->SetWindowMouseGrab = X11_SetWindowMouseGrab;
    device->SetWindowKeyboardGrab = X11_SetWindowKeyboardGrab;
    device->DestroyWindow = X11_DestroyWindow;
    device->CreateWindowFramebuffer = X11_CreateWindowFramebuffer;
    device->UpdateWindowFramebuffer = X11_UpdateWindowFramebuffer;
    device->DestroyWindowFramebuffer = X11_DestroyWindowFramebuffer;
    device->SetWindowHitTest = X11_SetWindowHitTest;
    device->AcceptDragAndDrop = X11_AcceptDragAndDrop;
    device->UpdateWindowShape = X11_UpdateWindowShape;
    device->FlashWindow = X11_FlashWindow;
    device->ShowWindowSystemMenu = X11_ShowWindowSystemMenu;
    device->SetWindowFocusable = X11_SetWindowFocusable;
    device->SyncWindow = X11_SyncWindow;

#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
    device->SetWindowMouseRect = X11_SetWindowMouseRect;
#endif // SDL_VIDEO_DRIVER_X11_XFIXES

#ifdef SDL_VIDEO_OPENGL_GLX
    device->GL_LoadLibrary = X11_GL_LoadLibrary;
    device->GL_GetProcAddress = X11_GL_GetProcAddress;
    device->GL_UnloadLibrary = X11_GL_UnloadLibrary;
    device->GL_CreateContext = X11_GL_CreateContext;
    device->GL_MakeCurrent = X11_GL_MakeCurrent;
    device->GL_SetSwapInterval = X11_GL_SetSwapInterval;
    device->GL_GetSwapInterval = X11_GL_GetSwapInterval;
    device->GL_SwapWindow = X11_GL_SwapWindow;
    device->GL_DestroyContext = X11_GL_DestroyContext;
    device->GL_GetEGLSurface = NULL;
#endif
#ifdef SDL_VIDEO_OPENGL_EGL
#ifdef SDL_VIDEO_OPENGL_GLX
    if (SDL_GetHintBoolean(SDL_HINT_VIDEO_FORCE_EGL, false)) {
#endif
        device->GL_LoadLibrary = X11_GLES_LoadLibrary;
        device->GL_GetProcAddress = X11_GLES_GetProcAddress;
        device->GL_UnloadLibrary = X11_GLES_UnloadLibrary;
        device->GL_CreateContext = X11_GLES_CreateContext;
        device->GL_MakeCurrent = X11_GLES_MakeCurrent;
        device->GL_SetSwapInterval = X11_GLES_SetSwapInterval;
        device->GL_GetSwapInterval = X11_GLES_GetSwapInterval;
        device->GL_SwapWindow = X11_GLES_SwapWindow;
        device->GL_DestroyContext = X11_GLES_DestroyContext;
        device->GL_GetEGLSurface = X11_GLES_GetEGLSurface;
#ifdef SDL_VIDEO_OPENGL_GLX
    }
#endif
#endif

    device->GetTextMimeTypes = X11_GetTextMimeTypes;
    device->SetClipboardData = X11_SetClipboardData;
    device->GetClipboardData = X11_GetClipboardData;
    device->HasClipboardData = X11_HasClipboardData;
    device->SetPrimarySelectionText = X11_SetPrimarySelectionText;
    device->GetPrimarySelectionText = X11_GetPrimarySelectionText;
    device->HasPrimarySelectionText = X11_HasPrimarySelectionText;
    device->StartTextInput = X11_StartTextInput;
    device->StopTextInput = X11_StopTextInput;
    device->UpdateTextInputArea = X11_UpdateTextInputArea;
    device->HasScreenKeyboardSupport = X11_HasScreenKeyboardSupport;
    device->ShowScreenKeyboard = X11_ShowScreenKeyboard;
    device->HideScreenKeyboard = X11_HideScreenKeyboard;
    device->IsScreenKeyboardShown = X11_IsScreenKeyboardShown;

    device->free = X11_DeleteDevice;

#ifdef SDL_VIDEO_VULKAN
    device->Vulkan_LoadLibrary = X11_Vulkan_LoadLibrary;
    device->Vulkan_UnloadLibrary = X11_Vulkan_UnloadLibrary;
    device->Vulkan_GetInstanceExtensions = X11_Vulkan_GetInstanceExtensions;
    device->Vulkan_CreateSurface = X11_Vulkan_CreateSurface;
    device->Vulkan_DestroySurface = X11_Vulkan_DestroySurface;
    device->Vulkan_GetPresentationSupport = X11_Vulkan_GetPresentationSupport;
#endif

#ifdef SDL_USE_LIBDBUS
    if (SDL_SystemTheme_Init())
        device->system_theme = SDL_SystemTheme_Get();
#endif

    device->device_caps = VIDEO_DEVICE_CAPS_HAS_POPUP_WINDOW_SUPPORT;

    /* Openbox doesn't send the new window dimensions when entering fullscreen, so the events must be synthesized.
     * This is otherwise not wanted, as it can break fullscreen window positioning on multi-monitor configurations.
     */
    if (!X11_CheckCurrentDesktop("openbox")) {
        device->device_caps |= VIDEO_DEVICE_CAPS_SENDS_DISPLAY_CHANGES;
    }

    data->is_xwayland = X11_IsXWayland(x11_display);
    if (data->is_xwayland) {
        SDL_LogInfo(SDL_LOG_CATEGORY_VIDEO, "Detected XWayland");

        device->device_caps |= VIDEO_DEVICE_CAPS_MODE_SWITCHING_EMULATED |
                               VIDEO_DEVICE_CAPS_DISABLE_MOUSE_WARP_ON_FULLSCREEN_TRANSITIONS;
    }

    return device;
}

VideoBootStrap X11_bootstrap = {
    "x11", "SDL X11 video driver",
    X11_CreateDevice,
    X11_ShowMessageBox,
    false
};

static int (*handler)(Display *, XErrorEvent *) = NULL;
static int X11_CheckWindowManagerErrorHandler(Display *d, XErrorEvent *e)
{
    if (e->error_code == BadWindow) {
        return 0;
    } else {
        return handler(d, e);
    }
}

static void X11_CheckWindowManager(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;
    Display *display = data->display;
    Atom _NET_SUPPORTING_WM_CHECK;
    int status, real_format;
    Atom real_type;
    unsigned long items_read = 0, items_left = 0;
    unsigned char *propdata = NULL;
    Window wm_window = 0;
#ifdef DEBUG_WINDOW_MANAGER
    char *wm_name;
#endif

    // Set up a handler to gracefully catch errors
    X11_XSync(display, False);
    handler = X11_XSetErrorHandler(X11_CheckWindowManagerErrorHandler);

    _NET_SUPPORTING_WM_CHECK = X11_XInternAtom(display, "_NET_SUPPORTING_WM_CHECK", False);
    status = X11_XGetWindowProperty(display, DefaultRootWindow(display), _NET_SUPPORTING_WM_CHECK, 0L, 1L, False, XA_WINDOW, &real_type, &real_format, &items_read, &items_left, &propdata);
    if (status == Success) {
        if (items_read) {
            wm_window = ((Window *)propdata)[0];
        }
        if (propdata) {
            X11_XFree(propdata);
            propdata = NULL;
        }
    }

    if (wm_window) {
        status = X11_XGetWindowProperty(display, wm_window, _NET_SUPPORTING_WM_CHECK, 0L, 1L, False, XA_WINDOW, &real_type, &real_format, &items_read, &items_left, &propdata);
        if (status != Success || !items_read || wm_window != ((Window *)propdata)[0]) {
            wm_window = None;
        }
        if (status == Success && propdata) {
            X11_XFree(propdata);
            propdata = NULL;
        }
    }

    // Reset the error handler, we're done checking
    X11_XSync(display, False);
    X11_XSetErrorHandler(handler);

    if (!wm_window) {
#ifdef DEBUG_WINDOW_MANAGER
        printf("Couldn't get _NET_SUPPORTING_WM_CHECK property\n");
#endif
        return;
    }
    data->net_wm = true;

#ifdef DEBUG_WINDOW_MANAGER
    wm_name = X11_GetWindowTitle(_this, wm_window);
    printf("Window manager: %s\n", wm_name);
    SDL_free(wm_name);
#endif
}

static bool X11_VideoInit(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;

    // Get the process PID to be associated to the window
    data->pid = getpid();

    // I have no idea how random this actually is, or has to be.
    data->window_group = (XID)(((size_t)data->pid) ^ ((size_t)_this));

    // Look up some useful Atoms
#define GET_ATOM(X) data->atoms.X = X11_XInternAtom(data->display, #X, False)
    GET_ATOM(WM_PROTOCOLS);
    GET_ATOM(WM_DELETE_WINDOW);
    GET_ATOM(WM_TAKE_FOCUS);
    GET_ATOM(WM_NAME);
    GET_ATOM(WM_TRANSIENT_FOR);
    GET_ATOM(_NET_WM_STATE);
    GET_ATOM(_NET_WM_STATE_HIDDEN);
    GET_ATOM(_NET_WM_STATE_FOCUSED);
    GET_ATOM(_NET_WM_STATE_MAXIMIZED_VERT);
    GET_ATOM(_NET_WM_STATE_MAXIMIZED_HORZ);
    GET_ATOM(_NET_WM_STATE_FULLSCREEN);
    GET_ATOM(_NET_WM_STATE_ABOVE);
    GET_ATOM(_NET_WM_STATE_SKIP_TASKBAR);
    GET_ATOM(_NET_WM_STATE_SKIP_PAGER);
    GET_ATOM(_NET_WM_MOVERESIZE);
    GET_ATOM(_NET_WM_STATE_MODAL);
    GET_ATOM(_NET_WM_ALLOWED_ACTIONS);
    GET_ATOM(_NET_WM_ACTION_FULLSCREEN);
    GET_ATOM(_NET_WM_NAME);
    GET_ATOM(_NET_WM_ICON_NAME);
    GET_ATOM(_NET_WM_ICON);
    GET_ATOM(_NET_WM_PING);
    GET_ATOM(_NET_WM_SYNC_REQUEST);
    GET_ATOM(_NET_WM_SYNC_REQUEST_COUNTER);
    GET_ATOM(_NET_WM_WINDOW_OPACITY);
    GET_ATOM(_NET_WM_USER_TIME);
    GET_ATOM(_NET_ACTIVE_WINDOW);
    GET_ATOM(_NET_FRAME_EXTENTS);
    GET_ATOM(_SDL_WAKEUP);
    GET_ATOM(UTF8_STRING);
    GET_ATOM(PRIMARY);
    GET_ATOM(CLIPBOARD);
    GET_ATOM(INCR);
    GET_ATOM(SDL_SELECTION);
    GET_ATOM(TARGETS);
    GET_ATOM(SDL_FORMATS);
    GET_ATOM(XdndAware);
    GET_ATOM(XdndEnter);
    GET_ATOM(XdndLeave);
    GET_ATOM(XdndPosition);
    GET_ATOM(XdndStatus);
    GET_ATOM(XdndTypeList);
    GET_ATOM(XdndActionCopy);
    GET_ATOM(XdndDrop);
    GET_ATOM(XdndFinished);
    GET_ATOM(XdndSelection);
    GET_ATOM(XKLAVIER_STATE);

    // Detect the window manager
    X11_CheckWindowManager(_this);

    if (!X11_InitModes(_this)) {
        return false;
    }

    if (!X11_InitXinput2(_this)) {
        // Assume a mouse and keyboard are attached
        SDL_AddKeyboard(SDL_DEFAULT_KEYBOARD_ID, NULL, false);
        SDL_AddMouse(SDL_DEFAULT_MOUSE_ID, NULL, false);
    }

#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
    X11_InitXfixes(_this);
#endif

    X11_InitXsettings(_this);

#ifdef SDL_VIDEO_DRIVER_X11_XSYNC
    X11_InitXsync(_this);
#endif

#ifdef SDL_VIDEO_DRIVER_X11_XTEST
    X11_InitXTest(_this);
#endif

#ifndef X_HAVE_UTF8_STRING
#warning X server does not support UTF8_STRING, a feature introduced in 2000! This is likely to become a hard error in a future libSDL3.
#endif

    if (!X11_InitKeyboard(_this)) {
        return false;
    }
    X11_InitMouse(_this);

    X11_InitTouch(_this);

    X11_InitPen(_this);

    return true;
}

void X11_VideoQuit(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;

    if (data->clipboard_window) {
        X11_XDestroyWindow(data->display, data->clipboard_window);
    }

    if (data->xsettings_window) {
        X11_XDestroyWindow(data->display, data->xsettings_window);
    }

#ifdef X_HAVE_UTF8_STRING
    if (data->im) {
        X11_XCloseIM(data->im);
    }
#endif

    X11_QuitModes(_this);
    X11_QuitKeyboard(_this);
    X11_QuitMouse(_this);
    X11_QuitTouch(_this);
    X11_QuitPen(_this);
    X11_QuitClipboard(_this);
    X11_QuitXsettings(_this);
}

bool X11_UseDirectColorVisuals(void)
{
    if (SDL_GetHintBoolean(SDL_HINT_VIDEO_X11_NODIRECTCOLOR, false)) {
        return false;
    }
    return true;
}

#endif // SDL_VIDEO_DRIVER_X11
