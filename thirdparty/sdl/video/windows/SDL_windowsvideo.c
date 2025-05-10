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

#ifdef SDL_VIDEO_DRIVER_WINDOWS

#ifdef SDL_VIDEO_VULKAN
#include "../SDL_vulkan_internal.h"
#endif
#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../SDL_hints_c.h"
#include "../../core/windows/SDL_hid.h"

#include "SDL_windowsvideo.h"
#include "SDL_windowsframebuffer.h"
#include "SDL_windowsmessagebox.h"
#include "SDL_windowsrawinput.h"
#include "SDL_windowsvulkan.h"

#ifdef HAVE_SHOBJIDL_CORE_H
#include <shobjidl_core.h>
#endif

#ifdef SDL_GDK_TEXTINPUT
#include "../gdk/SDL_gdktextinput.h"
#endif

// #define HIGHDPI_DEBUG

// Initialization/Query functions
static bool WIN_VideoInit(SDL_VideoDevice *_this);
static void WIN_VideoQuit(SDL_VideoDevice *_this);

// Hints
bool g_WindowsEnableMessageLoop = true;
bool g_WindowsEnableMenuMnemonics = false;
bool g_WindowFrameUsableWhileCursorHidden = true;

static void SDLCALL UpdateWindowsRawKeyboard(void *userdata, const char *name, const char *oldValue, const char *newValue)
{
    SDL_VideoDevice *_this = (SDL_VideoDevice *)userdata;
    bool enabled = SDL_GetStringBoolean(newValue, false);
    WIN_SetRawKeyboardEnabled(_this, enabled);
}

static void SDLCALL UpdateWindowsEnableMessageLoop(void *userdata, const char *name, const char *oldValue, const char *newValue)
{
    g_WindowsEnableMessageLoop = SDL_GetStringBoolean(newValue, true);
}

static void SDLCALL UpdateWindowsEnableMenuMnemonics(void *userdata, const char *name, const char *oldValue, const char *newValue)
{
    g_WindowsEnableMenuMnemonics = SDL_GetStringBoolean(newValue, false);
}

static void SDLCALL UpdateWindowFrameUsableWhileCursorHidden(void *userdata, const char *name, const char *oldValue, const char *newValue)
{
    g_WindowFrameUsableWhileCursorHidden = SDL_GetStringBoolean(newValue, true);
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
static bool WIN_SuspendScreenSaver(SDL_VideoDevice *_this)
{
    DWORD result;
    if (_this->suspend_screensaver) {
        result = SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED);
    } else {
        result = SetThreadExecutionState(ES_CONTINUOUS);
    }
    if (result == 0) {
        SDL_SetError("SetThreadExecutionState() failed");
        return false;
    }
    return true;
}
#endif

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
extern void D3D12_XBOX_GetResolution(Uint32 *width, Uint32 *height);
#endif

// Windows driver bootstrap functions

static void WIN_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_VideoData *data = device->internal;

    SDL_UnregisterApp();
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (data->userDLL) {
        SDL_UnloadObject(data->userDLL);
    }
    if (data->shcoreDLL) {
        SDL_UnloadObject(data->shcoreDLL);
    }
    if (data->dwmapiDLL) {
        SDL_UnloadObject(data->dwmapiDLL);
    }
#endif
#ifdef HAVE_DXGI_H
    if (data->pDXGIFactory) {
        IDXGIFactory_Release(data->pDXGIFactory);
    }
    if (data->dxgiDLL) {
        SDL_UnloadObject(data->dxgiDLL);
    }
#endif
    if (device->wakeup_lock) {
        SDL_DestroyMutex(device->wakeup_lock);
    }
    SDL_free(device->internal->rawinput);
    SDL_free(device->internal);
    SDL_free(device);
}

static SDL_VideoDevice *WIN_CreateDevice(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *data;

    SDL_RegisterApp(NULL, 0, NULL);

    // Initialize all variables that we clean on shutdown
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (device) {
        data = (SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    } else {
        data = NULL;
    }
    if (!data) {
        SDL_UnregisterApp();
        SDL_free(device);
        return NULL;
    }
    device->internal = data;
    device->wakeup_lock = SDL_CreateMutex();
    device->system_theme = WIN_GetSystemTheme();

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    data->userDLL = SDL_LoadObject("USER32.DLL");
    if (data->userDLL) {
        /* *INDENT-OFF* */ // clang-format off
        data->CloseTouchInputHandle = (BOOL (WINAPI *)(HTOUCHINPUT))SDL_LoadFunction(data->userDLL, "CloseTouchInputHandle");
        data->GetTouchInputInfo = (BOOL (WINAPI *)(HTOUCHINPUT, UINT, PTOUCHINPUT, int)) SDL_LoadFunction(data->userDLL, "GetTouchInputInfo");
        data->RegisterTouchWindow = (BOOL (WINAPI *)(HWND, ULONG))SDL_LoadFunction(data->userDLL, "RegisterTouchWindow");
        data->SetProcessDPIAware = (BOOL (WINAPI *)(void))SDL_LoadFunction(data->userDLL, "SetProcessDPIAware");
        data->SetProcessDpiAwarenessContext = (BOOL (WINAPI *)(DPI_AWARENESS_CONTEXT))SDL_LoadFunction(data->userDLL, "SetProcessDpiAwarenessContext");
        data->SetThreadDpiAwarenessContext = (DPI_AWARENESS_CONTEXT (WINAPI *)(DPI_AWARENESS_CONTEXT))SDL_LoadFunction(data->userDLL, "SetThreadDpiAwarenessContext");
        data->GetThreadDpiAwarenessContext = (DPI_AWARENESS_CONTEXT (WINAPI *)(void))SDL_LoadFunction(data->userDLL, "GetThreadDpiAwarenessContext");
        data->GetAwarenessFromDpiAwarenessContext = (DPI_AWARENESS (WINAPI *)(DPI_AWARENESS_CONTEXT))SDL_LoadFunction(data->userDLL, "GetAwarenessFromDpiAwarenessContext");
        data->EnableNonClientDpiScaling = (BOOL (WINAPI *)(HWND))SDL_LoadFunction(data->userDLL, "EnableNonClientDpiScaling");
        data->AdjustWindowRectExForDpi = (BOOL (WINAPI *)(LPRECT, DWORD, BOOL, DWORD, UINT))SDL_LoadFunction(data->userDLL, "AdjustWindowRectExForDpi");
        data->GetDpiForWindow = (UINT (WINAPI *)(HWND))SDL_LoadFunction(data->userDLL, "GetDpiForWindow");
        data->AreDpiAwarenessContextsEqual = (BOOL (WINAPI *)(DPI_AWARENESS_CONTEXT, DPI_AWARENESS_CONTEXT))SDL_LoadFunction(data->userDLL, "AreDpiAwarenessContextsEqual");
        data->IsValidDpiAwarenessContext = (BOOL (WINAPI *)(DPI_AWARENESS_CONTEXT))SDL_LoadFunction(data->userDLL, "IsValidDpiAwarenessContext");
        data->GetDisplayConfigBufferSizes = (LONG (WINAPI *)(UINT32,UINT32*,UINT32* ))SDL_LoadFunction(data->userDLL, "GetDisplayConfigBufferSizes");
        data->QueryDisplayConfig = (LONG (WINAPI *)(UINT32,UINT32*,DISPLAYCONFIG_PATH_INFO*,UINT32*,DISPLAYCONFIG_MODE_INFO*,DISPLAYCONFIG_TOPOLOGY_ID*))SDL_LoadFunction(data->userDLL, "QueryDisplayConfig");
        data->DisplayConfigGetDeviceInfo = (LONG (WINAPI *)(DISPLAYCONFIG_DEVICE_INFO_HEADER*))SDL_LoadFunction(data->userDLL, "DisplayConfigGetDeviceInfo");
        data->GetPointerType = (BOOL (WINAPI *)(UINT32, POINTER_INPUT_TYPE *))SDL_LoadFunction(data->userDLL, "GetPointerType");
        data->GetPointerPenInfo = (BOOL (WINAPI *)(UINT32, POINTER_PEN_INFO *))SDL_LoadFunction(data->userDLL, "GetPointerPenInfo");
        /* *INDENT-ON* */ // clang-format on
    } else {
        SDL_ClearError();
    }

    data->shcoreDLL = SDL_LoadObject("SHCORE.DLL");
    if (data->shcoreDLL) {
        /* *INDENT-OFF* */ // clang-format off
        data->GetDpiForMonitor = (HRESULT (WINAPI *)(HMONITOR, MONITOR_DPI_TYPE, UINT *, UINT *))SDL_LoadFunction(data->shcoreDLL, "GetDpiForMonitor");
        data->SetProcessDpiAwareness = (HRESULT (WINAPI *)(PROCESS_DPI_AWARENESS))SDL_LoadFunction(data->shcoreDLL, "SetProcessDpiAwareness");
        /* *INDENT-ON* */ // clang-format on
    } else {
        SDL_ClearError();
    }

    data->dwmapiDLL = SDL_LoadObject("DWMAPI.DLL");
    if (data->dwmapiDLL) {
        /* *INDENT-OFF* */ // clang-format off
        data->DwmFlush = (HRESULT (WINAPI *)(void))SDL_LoadFunction(data->dwmapiDLL, "DwmFlush");
        data->DwmEnableBlurBehindWindow = (HRESULT (WINAPI *)(HWND hwnd, const DWM_BLURBEHIND *pBlurBehind))SDL_LoadFunction(data->dwmapiDLL, "DwmEnableBlurBehindWindow");
        data->DwmSetWindowAttribute = (HRESULT (WINAPI *)(HWND hwnd, DWORD dwAttribute, LPCVOID pvAttribute, DWORD cbAttribute))SDL_LoadFunction(data->dwmapiDLL, "DwmSetWindowAttribute");
        /* *INDENT-ON* */ // clang-format on
    } else {
        SDL_ClearError();
    }
#endif // #if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

#ifdef HAVE_DXGI_H
    data->dxgiDLL = SDL_LoadObject("DXGI.DLL");
    if (data->dxgiDLL) {
        /* *INDENT-OFF* */ // clang-format off
        typedef HRESULT (WINAPI *CreateDXGI_t)(REFIID riid, void **ppFactory);
        /* *INDENT-ON* */ // clang-format on
        CreateDXGI_t CreateDXGI;

        CreateDXGI = (CreateDXGI_t)SDL_LoadFunction(data->dxgiDLL, "CreateDXGIFactory");
        if (CreateDXGI) {
            GUID dxgiGUID = { 0x7b7166ec, 0x21c7, 0x44ae, { 0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69 } };
            CreateDXGI(&dxgiGUID, (void **)&data->pDXGIFactory);
        }
    }
#endif

    // Set the function pointers
    device->VideoInit = WIN_VideoInit;
    device->VideoQuit = WIN_VideoQuit;
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    device->RefreshDisplays = WIN_RefreshDisplays;
    device->GetDisplayBounds = WIN_GetDisplayBounds;
    device->GetDisplayUsableBounds = WIN_GetDisplayUsableBounds;
    device->GetDisplayModes = WIN_GetDisplayModes;
    device->SetDisplayMode = WIN_SetDisplayMode;
#endif
    device->PumpEvents = WIN_PumpEvents;
    device->WaitEventTimeout = WIN_WaitEventTimeout;
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    device->SendWakeupEvent = WIN_SendWakeupEvent;
    device->SuspendScreenSaver = WIN_SuspendScreenSaver;
#endif

    device->CreateSDLWindow = WIN_CreateWindow;
    device->SetWindowTitle = WIN_SetWindowTitle;
    device->SetWindowIcon = WIN_SetWindowIcon;
    device->SetWindowPosition = WIN_SetWindowPosition;
    device->SetWindowSize = WIN_SetWindowSize;
    device->GetWindowBordersSize = WIN_GetWindowBordersSize;
    device->GetWindowSizeInPixels = WIN_GetWindowSizeInPixels;
    device->SetWindowOpacity = WIN_SetWindowOpacity;
    device->ShowWindow = WIN_ShowWindow;
    device->HideWindow = WIN_HideWindow;
    device->RaiseWindow = WIN_RaiseWindow;
    device->MaximizeWindow = WIN_MaximizeWindow;
    device->MinimizeWindow = WIN_MinimizeWindow;
    device->RestoreWindow = WIN_RestoreWindow;
    device->SetWindowBordered = WIN_SetWindowBordered;
    device->SetWindowResizable = WIN_SetWindowResizable;
    device->SetWindowAlwaysOnTop = WIN_SetWindowAlwaysOnTop;
    device->SetWindowFullscreen = WIN_SetWindowFullscreen;
    device->SetWindowParent = WIN_SetWindowParent;
    device->SetWindowModal = WIN_SetWindowModal;
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    device->GetWindowICCProfile = WIN_GetWindowICCProfile;
    device->SetWindowMouseRect = WIN_SetWindowMouseRect;
    device->SetWindowMouseGrab = WIN_SetWindowMouseGrab;
    device->SetWindowKeyboardGrab = WIN_SetWindowKeyboardGrab;
#endif
    device->DestroyWindow = WIN_DestroyWindow;
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    device->CreateWindowFramebuffer = WIN_CreateWindowFramebuffer;
    device->UpdateWindowFramebuffer = WIN_UpdateWindowFramebuffer;
    device->DestroyWindowFramebuffer = WIN_DestroyWindowFramebuffer;
    device->OnWindowEnter = WIN_OnWindowEnter;
    device->SetWindowHitTest = WIN_SetWindowHitTest;
    device->AcceptDragAndDrop = WIN_AcceptDragAndDrop;
    device->FlashWindow = WIN_FlashWindow;
    device->ApplyWindowProgress = WIN_ApplyWindowProgress;
    device->ShowWindowSystemMenu = WIN_ShowWindowSystemMenu;
    device->SetWindowFocusable = WIN_SetWindowFocusable;
    device->UpdateWindowShape = WIN_UpdateWindowShape;
#endif

#ifdef SDL_VIDEO_OPENGL_WGL
    device->GL_LoadLibrary = WIN_GL_LoadLibrary;
    device->GL_GetProcAddress = WIN_GL_GetProcAddress;
    device->GL_UnloadLibrary = WIN_GL_UnloadLibrary;
    device->GL_CreateContext = WIN_GL_CreateContext;
    device->GL_MakeCurrent = WIN_GL_MakeCurrent;
    device->GL_SetSwapInterval = WIN_GL_SetSwapInterval;
    device->GL_GetSwapInterval = WIN_GL_GetSwapInterval;
    device->GL_SwapWindow = WIN_GL_SwapWindow;
    device->GL_DestroyContext = WIN_GL_DestroyContext;
    device->GL_GetEGLSurface = NULL;
#endif
#ifdef SDL_VIDEO_OPENGL_EGL
#ifdef SDL_VIDEO_OPENGL_WGL
    if (SDL_GetHintBoolean(SDL_HINT_VIDEO_FORCE_EGL, false)) {
#endif
        // Use EGL based functions
        device->GL_LoadLibrary = WIN_GLES_LoadLibrary;
        device->GL_GetProcAddress = WIN_GLES_GetProcAddress;
        device->GL_UnloadLibrary = WIN_GLES_UnloadLibrary;
        device->GL_CreateContext = WIN_GLES_CreateContext;
        device->GL_MakeCurrent = WIN_GLES_MakeCurrent;
        device->GL_SetSwapInterval = WIN_GLES_SetSwapInterval;
        device->GL_GetSwapInterval = WIN_GLES_GetSwapInterval;
        device->GL_SwapWindow = WIN_GLES_SwapWindow;
        device->GL_DestroyContext = WIN_GLES_DestroyContext;
        device->GL_GetEGLSurface = WIN_GLES_GetEGLSurface;
#ifdef SDL_VIDEO_OPENGL_WGL
    }
#endif
#endif
#ifdef SDL_VIDEO_VULKAN
    device->Vulkan_LoadLibrary = WIN_Vulkan_LoadLibrary;
    device->Vulkan_UnloadLibrary = WIN_Vulkan_UnloadLibrary;
    device->Vulkan_GetInstanceExtensions = WIN_Vulkan_GetInstanceExtensions;
    device->Vulkan_CreateSurface = WIN_Vulkan_CreateSurface;
    device->Vulkan_DestroySurface = WIN_Vulkan_DestroySurface;
    device->Vulkan_GetPresentationSupport = WIN_Vulkan_GetPresentationSupport;
#endif

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    device->StartTextInput = WIN_StartTextInput;
    device->StopTextInput = WIN_StopTextInput;
    device->UpdateTextInputArea = WIN_UpdateTextInputArea;
    device->ClearComposition = WIN_ClearComposition;

    device->SetClipboardData = WIN_SetClipboardData;
    device->GetClipboardData = WIN_GetClipboardData;
    device->HasClipboardData = WIN_HasClipboardData;
#endif

#ifdef SDL_GDK_TEXTINPUT
    GDK_EnsureHints();

    device->StartTextInput = GDK_StartTextInput;
    device->StopTextInput = GDK_StopTextInput;
    device->UpdateTextInputArea = GDK_UpdateTextInputArea;
    device->ClearComposition = GDK_ClearComposition;

    device->HasScreenKeyboardSupport = GDK_HasScreenKeyboardSupport;
    device->ShowScreenKeyboard = GDK_ShowScreenKeyboard;
    device->HideScreenKeyboard = GDK_HideScreenKeyboard;
    device->IsScreenKeyboardShown = GDK_IsScreenKeyboardShown;
#endif

    device->free = WIN_DeleteDevice;

    device->device_caps = VIDEO_DEVICE_CAPS_HAS_POPUP_WINDOW_SUPPORT |
                          VIDEO_DEVICE_CAPS_SENDS_FULLSCREEN_DIMENSIONS;

    return device;
}

VideoBootStrap WINDOWS_bootstrap = {
    "windows", "SDL Windows video driver", WIN_CreateDevice,
    #if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    WIN_ShowMessageBox,
    #else
    NULL,
    #endif
    false
};

static BOOL WIN_DeclareDPIAwareUnaware(SDL_VideoDevice *_this)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    SDL_VideoData *data = _this->internal;

    if (data->SetProcessDpiAwarenessContext) {
        return data->SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_UNAWARE);
    } else if (data->SetProcessDpiAwareness) {
        // Windows 8.1
        return SUCCEEDED(data->SetProcessDpiAwareness(PROCESS_DPI_UNAWARE));
    }
#endif
    return FALSE;
}

static BOOL WIN_DeclareDPIAwareSystem(SDL_VideoDevice *_this)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    SDL_VideoData *data = _this->internal;

    if (data->SetProcessDpiAwarenessContext) {
        // Windows 10, version 1607
        return data->SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_SYSTEM_AWARE);
    } else if (data->SetProcessDpiAwareness) {
        // Windows 8.1
        return SUCCEEDED(data->SetProcessDpiAwareness(PROCESS_SYSTEM_DPI_AWARE));
    } else if (data->SetProcessDPIAware) {
        // Windows Vista
        return data->SetProcessDPIAware();
    }
#endif
    return FALSE;
}

static BOOL WIN_DeclareDPIAwarePerMonitor(SDL_VideoDevice *_this)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    SDL_VideoData *data = _this->internal;

    if (data->SetProcessDpiAwarenessContext) {
        // Windows 10, version 1607
        return data->SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE);
    } else if (data->SetProcessDpiAwareness) {
        // Windows 8.1
        return SUCCEEDED(data->SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE));
    } else {
        // Older OS: fall back to system DPI aware
        return WIN_DeclareDPIAwareSystem(_this);
    }
#else
    return FALSE;
#endif
}

static BOOL WIN_DeclareDPIAwarePerMonitorV2(SDL_VideoDevice *_this)
{
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    return FALSE;
#else
    SDL_VideoData *data = _this->internal;

    // Declare DPI aware (may have been done in external code or a manifest, as well)
    if (data->SetProcessDpiAwarenessContext) {
        // Windows 10, version 1607

        /* NOTE: SetThreadDpiAwarenessContext doesn't work here with OpenGL - the OpenGL contents
           end up still getting OS scaled. (tested on Windows 10 21H1 19043.1348, NVIDIA 496.49)

           NOTE: Enabling DPI awareness through Windows Explorer
           (right click .exe -> Properties -> Compatibility -> High DPI Settings ->
           check "Override high DPI Scaling behaviour", select Application) gives
           a DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE context (at least on Windows 10 21H1), and
           setting DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 will fail.

           NOTE: Entering exclusive fullscreen in a DPI_AWARENESS_CONTEXT_UNAWARE process
           appears to cause Windows to change the .exe manifest to DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE
           on future launches. This means attempting to use DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
           will fail in the future until you manually clear the "Override high DPI Scaling behaviour"
           setting in Windows Explorer (tested on Windows 10 21H2).
         */
        if (data->SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
            return TRUE;
        } else {
            return WIN_DeclareDPIAwarePerMonitor(_this);
        }
    } else {
        // Older OS: fall back to per-monitor (or system)
        return WIN_DeclareDPIAwarePerMonitor(_this);
    }
#endif
}

#ifdef HIGHDPI_DEBUG
static const char *WIN_GetDPIAwareness(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;

    if (data->GetThreadDpiAwarenessContext && data->AreDpiAwarenessContextsEqual) {
        DPI_AWARENESS_CONTEXT context = data->GetThreadDpiAwarenessContext();

        if (data->AreDpiAwarenessContextsEqual(context, DPI_AWARENESS_CONTEXT_UNAWARE)) {
            return "unaware";
        } else if (data->AreDpiAwarenessContextsEqual(context, DPI_AWARENESS_CONTEXT_SYSTEM_AWARE)) {
            return "system";
        } else if (data->AreDpiAwarenessContextsEqual(context, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE)) {
            return "permonitor";
        } else if (data->AreDpiAwarenessContextsEqual(context, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
            return "permonitorv2";
        } else if (data->AreDpiAwarenessContextsEqual(context, DPI_AWARENESS_CONTEXT_UNAWARE_GDISCALED)) {
            return "unaware_gdiscaled";
        }
    }

    return "";
}
#endif

static void WIN_InitDPIAwareness(SDL_VideoDevice *_this)
{
    const char *hint = SDL_GetHint("SDL_WINDOWS_DPI_AWARENESS");

    if (!hint || SDL_strcmp(hint, "permonitorv2") == 0) {
        WIN_DeclareDPIAwarePerMonitorV2(_this);
    } else if (SDL_strcmp(hint, "permonitor") == 0) {
        WIN_DeclareDPIAwarePerMonitor(_this);
    } else if (SDL_strcmp(hint, "system") == 0) {
        WIN_DeclareDPIAwareSystem(_this);
    } else if (SDL_strcmp(hint, "unaware") == 0) {
        WIN_DeclareDPIAwareUnaware(_this);
    }
}

static bool WIN_VideoInit(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;
    HRESULT hr;

    hr = WIN_CoInitialize();
    if (SUCCEEDED(hr)) {
        data->coinitialized = true;

#if !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
        hr = OleInitialize(NULL);
        if (SUCCEEDED(hr)) {
            data->oleinitialized = true;
        } else {
            SDL_LogInfo(SDL_LOG_CATEGORY_VIDEO, "OleInitialize() failed: 0x%.8x, using fallback drag-n-drop functionality", (unsigned int)hr);
        }
#endif // !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))
    } else {
        SDL_LogInfo(SDL_LOG_CATEGORY_VIDEO, "CoInitialize() failed: 0x%.8x, using fallback drag-n-drop functionality", (unsigned int)hr);
    }

    WIN_InitDPIAwareness(_this);

#ifdef HIGHDPI_DEBUG
    SDL_Log("DPI awareness: %s", WIN_GetDPIAwareness(_this));
#endif

    if (SDL_GetHintBoolean(SDL_HINT_WINDOWS_GAMEINPUT, true)) {
        WIN_InitGameInput(_this);
    }

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    // For Xbox, we just need to create the single display
    {
        SDL_DisplayMode mode;

        SDL_zero(mode);
        D3D12_XBOX_GetResolution(&mode.w, &mode.h);
        mode.refresh_rate = 60.0f;
        mode.format = SDL_PIXELFORMAT_ARGB8888;

        SDL_AddBasicVideoDisplay(&mode);
    }
#else // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (!WIN_InitModes(_this)) {
        return false;
    }

    WIN_InitKeyboard(_this);
    WIN_InitMouse(_this);
    WIN_InitDeviceNotification();
    if (!_this->internal->gameinput_context) {
        WIN_CheckKeyboardAndMouseHotplug(_this, true);
    }
#endif

    SDL_AddHintCallback(SDL_HINT_WINDOWS_RAW_KEYBOARD, UpdateWindowsRawKeyboard, _this);
    SDL_AddHintCallback(SDL_HINT_WINDOWS_ENABLE_MESSAGELOOP, UpdateWindowsEnableMessageLoop, NULL);
    SDL_AddHintCallback(SDL_HINT_WINDOWS_ENABLE_MENU_MNEMONICS, UpdateWindowsEnableMenuMnemonics, NULL);
    SDL_AddHintCallback(SDL_HINT_WINDOW_FRAME_USABLE_WHILE_CURSOR_HIDDEN, UpdateWindowFrameUsableWhileCursorHidden, NULL);

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    data->_SDL_WAKEUP = RegisterWindowMessageA("_SDL_WAKEUP");
#endif
#if defined(HAVE_SHOBJIDL_CORE_H)
    data->WM_TASKBAR_BUTTON_CREATED = RegisterWindowMessageA("TaskbarButtonCreated");
#endif

    return true;
}

void WIN_VideoQuit(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;

    SDL_RemoveHintCallback(SDL_HINT_WINDOWS_RAW_KEYBOARD, UpdateWindowsRawKeyboard, _this);
    SDL_RemoveHintCallback(SDL_HINT_WINDOWS_ENABLE_MESSAGELOOP, UpdateWindowsEnableMessageLoop, NULL);
    SDL_RemoveHintCallback(SDL_HINT_WINDOWS_ENABLE_MENU_MNEMONICS, UpdateWindowsEnableMenuMnemonics, NULL);
    SDL_RemoveHintCallback(SDL_HINT_WINDOW_FRAME_USABLE_WHILE_CURSOR_HIDDEN, UpdateWindowFrameUsableWhileCursorHidden, NULL);

    WIN_SetRawMouseEnabled(_this, false);
    WIN_SetRawKeyboardEnabled(_this, false);
    WIN_QuitGameInput(_this);

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    WIN_QuitModes(_this);
    WIN_QuitDeviceNotification();
    WIN_QuitKeyboard(_this);
    WIN_QuitMouse(_this);

#if defined(HAVE_SHOBJIDL_CORE_H)
    if (data->taskbar_list) {
        IUnknown_Release(data->taskbar_list);
        data->taskbar_list = NULL;
    }
#endif

    if (data->oleinitialized) {
        OleUninitialize();
        data->oleinitialized = false;
    }
#endif // !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))

    if (data->coinitialized) {
        WIN_CoUninitialize();
        data->coinitialized = false;
    }
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
#define D3D_DEBUG_INFO
#include <d3d9.h>

#ifdef D3D_DEBUG_INFO
#ifndef D3D_SDK_VERSION
#define D3D_SDK_VERSION (32 | 0x80000000)
#endif
#ifndef D3D9b_SDK_VERSION
#define D3D9b_SDK_VERSION (31 | 0x80000000)
#endif
#else //
#ifndef D3D_SDK_VERSION
#define D3D_SDK_VERSION 32
#endif
#ifndef D3D9b_SDK_VERSION
#define D3D9b_SDK_VERSION 31
#endif
#endif

bool D3D_LoadDLL(void **pD3DDLL, IDirect3D9 **pDirect3D9Interface)
{
    *pD3DDLL = SDL_LoadObject("D3D9.DLL");
    if (*pD3DDLL) {
        /* *INDENT-OFF* */ // clang-format off
        typedef IDirect3D9 *(WINAPI *Direct3DCreate9_t)(UINT SDKVersion);
        typedef HRESULT (WINAPI* Direct3DCreate9Ex_t)(UINT SDKVersion, IDirect3D9Ex** ppD3D);
        /* *INDENT-ON* */ // clang-format on
        Direct3DCreate9_t Direct3DCreate9Func;

        if (SDL_GetHintBoolean(SDL_HINT_WINDOWS_USE_D3D9EX, false)) {
            Direct3DCreate9Ex_t Direct3DCreate9ExFunc;

            Direct3DCreate9ExFunc = (Direct3DCreate9Ex_t)SDL_LoadFunction(*pD3DDLL, "Direct3DCreate9Ex");
            if (Direct3DCreate9ExFunc) {
                IDirect3D9Ex *pDirect3D9ExInterface;
                HRESULT hr = Direct3DCreate9ExFunc(D3D_SDK_VERSION, &pDirect3D9ExInterface);
                if (SUCCEEDED(hr)) {
                    const GUID IDirect3D9_GUID = { 0x81bdcbca, 0x64d4, 0x426d, { 0xae, 0x8d, 0xad, 0x1, 0x47, 0xf4, 0x27, 0x5c } };
                    hr = IDirect3D9Ex_QueryInterface(pDirect3D9ExInterface, &IDirect3D9_GUID, (void **)pDirect3D9Interface);
                    IDirect3D9Ex_Release(pDirect3D9ExInterface);
                    if (SUCCEEDED(hr)) {
                        return true;
                    }
                }
            }
        }

        Direct3DCreate9Func = (Direct3DCreate9_t)SDL_LoadFunction(*pD3DDLL, "Direct3DCreate9");
        if (Direct3DCreate9Func) {
            *pDirect3D9Interface = Direct3DCreate9Func(D3D_SDK_VERSION);
            if (*pDirect3D9Interface) {
                return true;
            }
        }

        SDL_UnloadObject(*pD3DDLL);
        *pD3DDLL = NULL;
    }
    *pDirect3D9Interface = NULL;
    return false;
}

int SDL_GetDirect3D9AdapterIndex(SDL_DisplayID displayID)
{
    void *pD3DDLL;
    IDirect3D9 *pD3D;
    if (!D3D_LoadDLL(&pD3DDLL, &pD3D)) {
        SDL_SetError("Unable to create Direct3D interface");
        return -1;
    } else {
        SDL_DisplayData *pData = SDL_GetDisplayDriverData(displayID);
        int adapterIndex = D3DADAPTER_DEFAULT;

        if (!pData) {
            SDL_SetError("Invalid display index");
            adapterIndex = -1; // make sure we return something invalid
        } else {
            char *displayName = WIN_StringToUTF8W(pData->DeviceName);
            unsigned int count = IDirect3D9_GetAdapterCount(pD3D);
            unsigned int i;
            for (i = 0; i < count; i++) {
                D3DADAPTER_IDENTIFIER9 id;
                IDirect3D9_GetAdapterIdentifier(pD3D, i, 0, &id);

                if (SDL_strcmp(id.DeviceName, displayName) == 0) {
                    adapterIndex = i;
                    break;
                }
            }
            SDL_free(displayName);
        }

        // free up the D3D stuff we inited
        IDirect3D9_Release(pD3D);
        SDL_UnloadObject(pD3DDLL);

        return adapterIndex;
    }
}
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

bool SDL_GetDXGIOutputInfo(SDL_DisplayID displayID, int *adapterIndex, int *outputIndex)
{
#ifndef HAVE_DXGI_H
    if (adapterIndex) {
        *adapterIndex = -1;
    }
    if (outputIndex) {
        *outputIndex = -1;
    }
    return SDL_SetError("SDL was compiled without DXGI support due to missing dxgi.h header");
#else
    const SDL_VideoDevice *videodevice = SDL_GetVideoDevice();
    const SDL_VideoData *videodata = videodevice ? videodevice->internal : NULL;
    SDL_DisplayData *pData = SDL_GetDisplayDriverData(displayID);
    int nAdapter, nOutput;
    IDXGIAdapter *pDXGIAdapter;
    IDXGIOutput *pDXGIOutput;

    if (!adapterIndex) {
        return SDL_InvalidParamError("adapterIndex");
    }

    if (!outputIndex) {
        return SDL_InvalidParamError("outputIndex");
    }

    *adapterIndex = -1;
    *outputIndex = -1;

    if (!pData) {
        return SDL_SetError("Invalid display index");
    }

    if (!videodata || !videodata->pDXGIFactory) {
        return SDL_SetError("Unable to create DXGI interface");
    }

    nAdapter = 0;
    while (*adapterIndex == -1 && SUCCEEDED(IDXGIFactory_EnumAdapters(videodata->pDXGIFactory, nAdapter, &pDXGIAdapter))) {
        nOutput = 0;
        while (*adapterIndex == -1 && SUCCEEDED(IDXGIAdapter_EnumOutputs(pDXGIAdapter, nOutput, &pDXGIOutput))) {
            DXGI_OUTPUT_DESC outputDesc;
            if (SUCCEEDED(IDXGIOutput_GetDesc(pDXGIOutput, &outputDesc))) {
                if (SDL_wcscmp(outputDesc.DeviceName, pData->DeviceName) == 0) {
                    *adapterIndex = nAdapter;
                    *outputIndex = nOutput;
                }
            }
            IDXGIOutput_Release(pDXGIOutput);
            nOutput++;
        }
        IDXGIAdapter_Release(pDXGIAdapter);
        nAdapter++;
    }

    if (*adapterIndex == -1) {
        return SDL_SetError("Couldn't find matching adapter");
    }
    return true;
#endif
}

SDL_SystemTheme WIN_GetSystemTheme(void)
{
    SDL_SystemTheme theme = SDL_SYSTEM_THEME_LIGHT;
    HKEY hKey;
    DWORD dwType = REG_DWORD;
    DWORD value = ~0U;
    DWORD length = sizeof(value);

    // Technically this isn't the system theme, but it's the preference for applications
    if (RegOpenKeyExW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        if (RegQueryValueExW(hKey, L"AppsUseLightTheme", 0, &dwType, (LPBYTE)&value, &length) == ERROR_SUCCESS) {
            if (value == 0) {
                theme = SDL_SYSTEM_THEME_DARK;
            }
        }
        RegCloseKey(hKey);
    }
    return theme;
}

bool WIN_IsPerMonitorV2DPIAware(SDL_VideoDevice *_this)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    SDL_VideoData *data = _this->internal;

    if (data->AreDpiAwarenessContextsEqual && data->GetThreadDpiAwarenessContext) {
        // Windows 10, version 1607
        return data->AreDpiAwarenessContextsEqual(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, data->GetThreadDpiAwarenessContext());
    }
#endif
    return false;
}

#endif // SDL_VIDEO_DRIVER_WINDOWS
