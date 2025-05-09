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

#include "../../core/windows/SDL_windows.h"

#include "../../SDL_hints_c.h"
#include "../../events/SDL_dropevents_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_windowevents_c.h"
#include "../SDL_pixels_c.h"
#include "../SDL_sysvideo.h"

#include "SDL_windowsvideo.h"
#include "SDL_windowswindow.h"

// Dropfile support
#include <shellapi.h>

#ifdef HAVE_SHOBJIDL_CORE_H
#include <shobjidl_core.h>
#endif

// Dark mode support
typedef enum {
    UXTHEME_APPMODE_DEFAULT,
    UXTHEME_APPMODE_ALLOW_DARK,
    UXTHEME_APPMODE_FORCE_DARK,
    UXTHEME_APPMODE_FORCE_LIGHT,
    UXTHEME_APPMODE_MAX
} UxthemePreferredAppMode;

typedef enum {
    WCA_UNDEFINED = 0,
    WCA_USEDARKMODECOLORS = 26,
    WCA_LAST = 27
} WINDOWCOMPOSITIONATTRIB;

typedef struct {
    WINDOWCOMPOSITIONATTRIB Attrib;
    PVOID pvData;
    SIZE_T cbData;
} WINDOWCOMPOSITIONATTRIBDATA;

typedef struct {
    ULONG dwOSVersionInfoSize;
    ULONG dwMajorVersion;
    ULONG dwMinorVersion;
    ULONG dwBuildNumber;
    ULONG dwPlatformId;
    WCHAR szCSDVersion[128];
} NT_OSVERSIONINFOW;

typedef bool (WINAPI *ShouldAppsUseDarkMode_t)(void);
typedef void (WINAPI *AllowDarkModeForWindow_t)(HWND, bool);
typedef void (WINAPI *AllowDarkModeForApp_t)(bool);
typedef void (WINAPI *RefreshImmersiveColorPolicyState_t)(void);
typedef UxthemePreferredAppMode (WINAPI *SetPreferredAppMode_t)(UxthemePreferredAppMode);
typedef BOOL (WINAPI *SetWindowCompositionAttribute_t)(HWND, const WINDOWCOMPOSITIONATTRIBDATA *);
typedef void (NTAPI *RtlGetVersion_t)(NT_OSVERSIONINFOW *);

// Windows CE compatibility
#ifndef SWP_NOCOPYBITS
#define SWP_NOCOPYBITS 0
#endif

/* An undocumented message to create a popup system menu
 * - wParam is always 0
 * - lParam = MAKELONG(x, y) where x and y are the screen coordinates where the menu should be displayed
 */
#ifndef WM_POPUPSYSTEMMENU
#define WM_POPUPSYSTEMMENU 0x313
#endif

// #define HIGHDPI_DEBUG

// Fake window to help with DirectInput events.
HWND SDL_HelperWindow = NULL;
static const TCHAR *SDL_HelperWindowClassName = TEXT("SDLHelperWindowInputCatcher");
static const TCHAR *SDL_HelperWindowName = TEXT("SDLHelperWindowInputMsgWindow");
static ATOM SDL_HelperWindowClass = 0;

/* For borderless Windows, still want the following flag:
   - WS_MINIMIZEBOX: window will respond to Windows minimize commands sent to all windows, such as windows key + m, shaking title bar, etc.
   Additionally, non-fullscreen windows can add:
   - WS_CAPTION: this seems to enable the Windows minimize animation
   - WS_SYSMENU: enables system context menu on task bar
   This will also cause the task bar to overlap the window and other windowed behaviors, so only use this for windows that shouldn't appear to be fullscreen
   - WS_THICKFRAME: allows hit-testing to resize window (doesn't actually add a frame to a borderless window).
   - WS_MAXIMIZEBOX: window will respond to Windows maximize commands sent to all windows, and the window will fill the usable desktop area rather than the whole screen
 */

#define STYLE_BASIC               (WS_CLIPSIBLINGS | WS_CLIPCHILDREN)
#define STYLE_FULLSCREEN          (WS_POPUP | WS_MINIMIZEBOX)
#define STYLE_BORDERLESS          (WS_POPUP | WS_MINIMIZEBOX)
#define STYLE_BORDERLESS_WINDOWED (WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX)
#define STYLE_NORMAL              (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX)
#define STYLE_RESIZABLE           (WS_THICKFRAME | WS_MAXIMIZEBOX)
#define STYLE_MASK                (STYLE_FULLSCREEN | STYLE_BORDERLESS | STYLE_NORMAL | STYLE_RESIZABLE)

static DWORD GetWindowStyle(SDL_Window *window)
{
    DWORD style = 0;

    if (SDL_WINDOW_IS_POPUP(window)) {
        style |= WS_POPUP;
    } else if (window->flags & SDL_WINDOW_FULLSCREEN) {
        style |= STYLE_FULLSCREEN;
    } else {
        if (window->flags & SDL_WINDOW_BORDERLESS) {
            /* This behavior more closely matches other platform where the window is borderless
               but still interacts with the window manager (e.g. task bar shows above it, it can
               be resized to fit within usable desktop area, etc.)
             */
            if (SDL_GetHintBoolean("SDL_BORDERLESS_WINDOWED_STYLE", true)) {
                style |= STYLE_BORDERLESS_WINDOWED;
            } else {
                style |= STYLE_BORDERLESS;
            }
        } else {
            style |= STYLE_NORMAL;
        }

        /* The WS_MAXIMIZEBOX style flag needs to be retained for as long as the window is maximized,
         * or restoration from minimized can fail, and leaving maximized can result in an odd size.
         */
        if (window->flags & SDL_WINDOW_RESIZABLE) {
            /* You can have a borderless resizable window, but Windows doesn't always draw it correctly,
               see https://bugzilla.libsdl.org/show_bug.cgi?id=4466
             */
            if (!(window->flags & SDL_WINDOW_BORDERLESS) ||
                SDL_GetHintBoolean("SDL_BORDERLESS_RESIZABLE_STYLE", true)) {
                style |= STYLE_RESIZABLE;
            }
        }

        if (window->internal && window->internal->force_ws_maximizebox) {
            /* Even if the resizable flag is cleared, WS_MAXIMIZEBOX is still needed as long
             * as the window is maximized, or de-maximizing or minimizing and restoring the
             * maximized window can result in the window disappearing or being the wrong size.
             */
            style |= WS_MAXIMIZEBOX;
        }

        // Need to set initialize minimize style, or when we call ShowWindow with WS_MINIMIZE it will activate a random window
        if (window->flags & SDL_WINDOW_MINIMIZED) {
            style |= WS_MINIMIZE;
        }
    }
    return style;
}

static DWORD GetWindowStyleEx(SDL_Window *window)
{
    DWORD style = 0;

    if (SDL_WINDOW_IS_POPUP(window) || (window->flags & SDL_WINDOW_UTILITY)) {
        style |= WS_EX_TOOLWINDOW;
    }
    if (SDL_WINDOW_IS_POPUP(window) || (window->flags & SDL_WINDOW_NOT_FOCUSABLE)) {
        style |= WS_EX_NOACTIVATE;
    }
    return style;
}

#ifdef HAVE_SHOBJIDL_CORE_H
static ITaskbarList3 *GetTaskbarList(SDL_Window* window)
{
    const SDL_WindowData *data = window->internal;
    SDL_assert(data->taskbar_button_created);
    if (!data->videodata->taskbar_list) {
        HRESULT ret = CoCreateInstance(&CLSID_TaskbarList, NULL, CLSCTX_ALL, &IID_ITaskbarList3, (LPVOID *)&data->videodata->taskbar_list);
        if (FAILED(ret)) {
            WIN_SetErrorFromHRESULT("Unable to create taskbar list", ret);
            return NULL;
        }
        ITaskbarList3 *taskbarlist = data->videodata->taskbar_list;
        ret = taskbarlist->lpVtbl->HrInit(taskbarlist);
        if (FAILED(ret)) {
            taskbarlist->lpVtbl->Release(taskbarlist);
            data->videodata->taskbar_list = NULL;
            WIN_SetErrorFromHRESULT("Unable to initialize taskbar list", ret);
            return NULL;
        }
    }
    return data->videodata->taskbar_list;
}
#endif

/**
 * Returns arguments to pass to SetWindowPos - the window rect, including frame, in Windows coordinates.
 * Can be called before we have a HWND.
 */
static bool WIN_AdjustWindowRectWithStyle(SDL_Window *window, DWORD style, DWORD styleEx, BOOL menu, int *x, int *y, int *width, int *height, SDL_WindowRect rect_type)
{
    SDL_VideoData *videodata = SDL_GetVideoDevice() ? SDL_GetVideoDevice()->internal : NULL;
    RECT rect;

    // Client rect, in points
    switch (rect_type) {
    case SDL_WINDOWRECT_CURRENT:
        SDL_RelativeToGlobalForWindow(window, window->x, window->y, x, y);
        *width = window->w;
        *height = window->h;
        break;
    case SDL_WINDOWRECT_WINDOWED:
        SDL_RelativeToGlobalForWindow(window, window->windowed.x, window->windowed.y, x, y);
        *width = window->windowed.w;
        *height = window->windowed.h;
        break;
    case SDL_WINDOWRECT_FLOATING:
        SDL_RelativeToGlobalForWindow(window, window->floating.x, window->floating.y, x, y);
        *width = window->floating.w;
        *height = window->floating.h;
        break;
    case SDL_WINDOWRECT_PENDING:
        SDL_RelativeToGlobalForWindow(window, window->pending.x, window->pending.y, x, y);
        *width = window->pending.w;
        *height = window->pending.h;
        break;
    default:
        // Should never be here
        SDL_assert_release(false);
        *width = 0;
        *height = 0;
        break;
    }

    /* Copy the client size in pixels into this rect structure,
       which we'll then adjust with AdjustWindowRectEx */
    rect.left = 0;
    rect.top = 0;
    rect.right = *width;
    rect.bottom = *height;

    /* borderless windows will have WM_NCCALCSIZE return 0 for the non-client area. When this happens, it looks like windows will send a resize message
       expanding the window client area to the previous window + chrome size, so shouldn't need to adjust the window size for the set styles.
     */
    if (!(window->flags & SDL_WINDOW_BORDERLESS) && !SDL_WINDOW_IS_POPUP(window)) {
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
        AdjustWindowRectEx(&rect, style, menu, 0);
#else
        if (WIN_IsPerMonitorV2DPIAware(SDL_GetVideoDevice())) {
            /* With per-monitor v2, the window border/titlebar size depend on the DPI, so we need to call AdjustWindowRectExForDpi instead of
               AdjustWindowRectEx. */
            if (videodata) {
                UINT frame_dpi;
                SDL_WindowData *data = window->internal;
                frame_dpi = (data && videodata->GetDpiForWindow) ? videodata->GetDpiForWindow(data->hwnd) : USER_DEFAULT_SCREEN_DPI;
                if (videodata->AdjustWindowRectExForDpi(&rect, style, menu, styleEx, frame_dpi) == 0) {
                    return WIN_SetError("AdjustWindowRectExForDpi()");
                }
            }
        } else {
            if (AdjustWindowRectEx(&rect, style, menu, styleEx) == 0) {
                return WIN_SetError("AdjustWindowRectEx()");
            }
        }
#endif
    }

    // Final rect in Windows screen space, including the frame
    *x += rect.left;
    *y += rect.top;
    *width = (rect.right - rect.left);
    *height = (rect.bottom - rect.top);

#ifdef HIGHDPI_DEBUG
    SDL_Log("WIN_AdjustWindowRectWithStyle: in: %d, %d, %dx%d, returning: %d, %d, %dx%d, used dpi %d for frame calculation",
            (rect_type == SDL_WINDOWRECT_FLOATING ? window->floating.x : rect_type == SDL_WINDOWRECT_WINDOWED ? window->windowed.x : window->x),
            (rect_type == SDL_WINDOWRECT_FLOATING ? window->floating.y : rect_type == SDL_WINDOWRECT_WINDOWED ? window->windowed.y : window->y),
            (rect_type == SDL_WINDOWRECT_FLOATING ? window->floating.w : rect_type == SDL_WINDOWRECT_WINDOWED ? window->windowed.w : window->w),
            (rect_type == SDL_WINDOWRECT_FLOATING ? window->floating.h : rect_type == SDL_WINDOWRECT_WINDOWED ? window->windowed.h : window->h),
            *x, *y, *width, *height, frame_dpi);
#endif
    return true;
}

bool WIN_AdjustWindowRect(SDL_Window *window, int *x, int *y, int *width, int *height, SDL_WindowRect rect_type)
{
    SDL_WindowData *data = window->internal;
    HWND hwnd = data->hwnd;
    DWORD style, styleEx;
    BOOL menu;

    style = GetWindowLong(hwnd, GWL_STYLE);
    styleEx = GetWindowLong(hwnd, GWL_EXSTYLE);
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    menu = FALSE;
#else
    menu = (style & WS_CHILDWINDOW) ? FALSE : (GetMenu(hwnd) != NULL);
#endif
    return WIN_AdjustWindowRectWithStyle(window, style, styleEx, menu, x, y, width, height, rect_type);
}

bool WIN_AdjustWindowRectForHWND(HWND hwnd, LPRECT lpRect, UINT frame_dpi)
{
    SDL_VideoDevice *videodevice = SDL_GetVideoDevice();
    SDL_VideoData *videodata = videodevice ? videodevice->internal : NULL;
    DWORD style, styleEx;
    BOOL menu;

    style = GetWindowLong(hwnd, GWL_STYLE);
    styleEx = GetWindowLong(hwnd, GWL_EXSTYLE);
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    menu = FALSE;
#else
    menu = (style & WS_CHILDWINDOW) ? FALSE : (GetMenu(hwnd) != NULL);
#endif

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    AdjustWindowRectEx(lpRect, style, menu, styleEx);
#else
    if (WIN_IsPerMonitorV2DPIAware(videodevice)) {
        // With per-monitor v2, the window border/titlebar size depend on the DPI, so we need to call AdjustWindowRectExForDpi instead of AdjustWindowRectEx.
        if (!frame_dpi) {
            frame_dpi = videodata->GetDpiForWindow ? videodata->GetDpiForWindow(hwnd) : USER_DEFAULT_SCREEN_DPI;
        }
        if (!videodata->AdjustWindowRectExForDpi(lpRect, style, menu, styleEx, frame_dpi)) {
            return WIN_SetError("AdjustWindowRectExForDpi()");
        }
    } else {
        if (!AdjustWindowRectEx(lpRect, style, menu, styleEx)) {
            return WIN_SetError("AdjustWindowRectEx()");
        }
    }
#endif
    return true;
}

bool WIN_SetWindowPositionInternal(SDL_Window *window, UINT flags, SDL_WindowRect rect_type)
{
    SDL_Window *child_window;
    SDL_WindowData *data = window->internal;
    HWND hwnd = data->hwnd;
    HWND top;
    int x, y;
    int w, h;
    bool result = true;

    // Figure out what the window area will be
    if (SDL_ShouldAllowTopmost() && (window->flags & SDL_WINDOW_ALWAYS_ON_TOP)) {
        top = HWND_TOPMOST;
    } else {
        top = HWND_NOTOPMOST;
    }

    WIN_AdjustWindowRect(window, &x, &y, &w, &h, rect_type);

    data->expected_resize = true;
    if (SetWindowPos(hwnd, top, x, y, w, h, flags) == 0) {
        result = WIN_SetError("SetWindowPos()");
    }
    data->expected_resize = false;

    // Update any child windows
    for (child_window = window->first_child; child_window; child_window = child_window->next_sibling) {
        if (!child_window->internal) {
            // This child window is not yet fully initialized.
            continue;
        }
        if (!WIN_SetWindowPositionInternal(child_window, flags, SDL_WINDOWRECT_CURRENT)) {
            result = false;
        }
    }
    return result;
}

static SDL_WindowEraseBackgroundMode GetEraseBackgroundModeHint(void)
{
    const char *hint = SDL_GetHint(SDL_HINT_WINDOWS_ERASE_BACKGROUND_MODE);
    if (!hint)
        return SDL_ERASEBACKGROUNDMODE_INITIAL;

    if (SDL_strstr(hint, "never"))
        return SDL_ERASEBACKGROUNDMODE_NEVER;

    if (SDL_strstr(hint, "initial"))
        return SDL_ERASEBACKGROUNDMODE_INITIAL;

    if (SDL_strstr(hint, "always"))
        return SDL_ERASEBACKGROUNDMODE_ALWAYS;

    int mode = SDL_GetStringInteger(hint, 1);
    if (mode < 0 || mode > 2) {
        SDL_Log("GetEraseBackgroundModeHint: invalid value for SDL_HINT_WINDOWS_ERASE_BACKGROUND_MODE. Fallback to default");
        return SDL_ERASEBACKGROUNDMODE_INITIAL;
    }

    return (SDL_WindowEraseBackgroundMode)mode;
}

static bool SetupWindowData(SDL_VideoDevice *_this, SDL_Window *window, HWND hwnd, HWND parent)
{
    SDL_VideoData *videodata = _this->internal;
    SDL_WindowData *data;

    // Allocate the window data
    data = (SDL_WindowData *)SDL_calloc(1, sizeof(*data));
    if (!data) {
        return false;
    }
    data->window = window;
    data->hwnd = hwnd;
    data->parent = parent;
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    data->hdc = (HDC)data->hwnd;
#else
    data->hdc = GetDC(hwnd);
#endif
    data->hinstance = (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE);
    data->mouse_button_flags = (WPARAM)-1;
    data->last_pointer_update = (LPARAM)-1;
    data->videodata = videodata;
    data->initializing = true;
    data->last_displayID = window->last_displayID;
    data->hint_erase_background_mode = GetEraseBackgroundModeHint();


    // WIN_WarpCursor() jitters by +1, and remote desktop warp wobble is +/- 1
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    LONG remote_desktop_adjustment = GetSystemMetrics(SM_REMOTESESSION) ? 2 : 0;
    data->cursor_ctrlock_rect.left   = 0 - remote_desktop_adjustment;
    data->cursor_ctrlock_rect.top    = 0;
    data->cursor_ctrlock_rect.right  = 1 + remote_desktop_adjustment;
    data->cursor_ctrlock_rect.bottom = 1;
#endif

    if (SDL_GetHintBoolean("SDL_WINDOW_RETAIN_CONTENT", false)) {
        data->copybits_flag = 0;
    } else {
        data->copybits_flag = SWP_NOCOPYBITS;
    }

#ifdef HIGHDPI_DEBUG
    SDL_Log("SetupWindowData: initialized data->scaling_dpi to %d", data->scaling_dpi);
#endif

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    // Associate the data with the window
    if (!SetProp(hwnd, TEXT("SDL_WindowData"), data)) {
        ReleaseDC(hwnd, data->hdc);
        SDL_free(data);
        return WIN_SetError("SetProp() failed");
    }
#endif

    window->internal = data;

    // Set up the window proc function
#ifdef GWLP_WNDPROC
    data->wndproc = (WNDPROC)GetWindowLongPtr(hwnd, GWLP_WNDPROC);
    if (data->wndproc == WIN_WindowProc) {
        data->wndproc = NULL;
    } else {
        SetWindowLongPtr(hwnd, GWLP_WNDPROC, (LONG_PTR)WIN_WindowProc);
    }
#else
    data->wndproc = (WNDPROC)GetWindowLong(hwnd, GWL_WNDPROC);
    if (data->wndproc == WIN_WindowProc) {
        data->wndproc = NULL;
    } else {
        SetWindowLong(hwnd, GWL_WNDPROC, (LONG_PTR)WIN_WindowProc);
    }
#endif

    // Fill in the SDL window with the window state
    {
        DWORD style = GetWindowLong(hwnd, GWL_STYLE);
        if (style & WS_VISIBLE) {
            window->flags &= ~SDL_WINDOW_HIDDEN;
        } else {
            window->flags |= SDL_WINDOW_HIDDEN;
        }
        if (style & WS_POPUP) {
            window->flags |= SDL_WINDOW_BORDERLESS;
        } else {
            window->flags &= ~SDL_WINDOW_BORDERLESS;
        }
        if (style & WS_THICKFRAME) {
            window->flags |= SDL_WINDOW_RESIZABLE;
        } else if (!(style & WS_POPUP)) {
            window->flags &= ~SDL_WINDOW_RESIZABLE;
        }
#ifdef WS_MAXIMIZE
        if (style & WS_MAXIMIZE) {
            window->flags |= SDL_WINDOW_MAXIMIZED;
        } else
#endif
        {
            window->flags &= ~SDL_WINDOW_MAXIMIZED;
        }
#ifdef WS_MINIMIZE
        if (style & WS_MINIMIZE) {
            window->flags |= SDL_WINDOW_MINIMIZED;
        } else
#endif
        {
            window->flags &= ~SDL_WINDOW_MINIMIZED;
        }
    }
    if (!(window->flags & SDL_WINDOW_MINIMIZED)) {
        RECT rect;
        if (GetClientRect(hwnd, &rect) && WIN_WindowRectValid(&rect)) {
            int w = rect.right;
            int h = rect.bottom;

            if (window->flags & SDL_WINDOW_EXTERNAL) {
                window->floating.w = window->windowed.w = window->w = w;
                window->floating.h = window->windowed.h = window->h = h;
            } else if ((window->windowed.w && window->windowed.w != w) || (window->windowed.h && window->windowed.h != h)) {
                // We tried to create a window larger than the desktop and Windows didn't allow it.  Override!
                int x, y;
                // Figure out what the window area will be
                WIN_AdjustWindowRect(window, &x, &y, &w, &h, SDL_WINDOWRECT_FLOATING);
                data->expected_resize = true;
                SetWindowPos(hwnd, NULL, x, y, w, h, data->copybits_flag | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_NOACTIVATE);
                data->expected_resize = false;
            } else {
                window->w = w;
                window->h = h;
            }
        }
    }
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (!(window->flags & SDL_WINDOW_MINIMIZED)) {
        POINT point;
        point.x = 0;
        point.y = 0;
        if (ClientToScreen(hwnd, &point)) {
            if (window->flags & SDL_WINDOW_EXTERNAL) {
                window->floating.x = window->windowed.x = point.x;
                window->floating.y = window->windowed.y = point.y;
            }
            window->x = point.x;
            window->y = point.y;
        }
    }

    WIN_UpdateWindowICCProfile(window, false);
#endif

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    window->flags |= SDL_WINDOW_INPUT_FOCUS;
    SDL_SetKeyboardFocus(window);
#else
    if (GetFocus() == hwnd) {
        window->flags |= SDL_WINDOW_INPUT_FOCUS;
        SDL_SetKeyboardFocus(window);
        WIN_UpdateClipCursor(window);
    }
#endif

    if (window->flags & SDL_WINDOW_ALWAYS_ON_TOP) {
        WIN_SetWindowAlwaysOnTop(_this, window, true);
    } else {
        WIN_SetWindowAlwaysOnTop(_this, window, false);
    }

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    // Enable multi-touch
    if (videodata->RegisterTouchWindow) {
        videodata->RegisterTouchWindow(hwnd, (TWF_FINETOUCH | TWF_WANTPALM));
    }
#endif

    if (data->parent && !window->parent) {
        data->destroy_parent_with_window = true;
    }

    data->initializing = false;

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (window->flags & SDL_WINDOW_EXTERNAL) {
        // Query the title from the existing window
        LPTSTR title;
        int titleLen;
        bool isstack;

        titleLen = GetWindowTextLength(hwnd);
        title = SDL_small_alloc(TCHAR, titleLen + 1, &isstack);
        if (title) {
            titleLen = GetWindowText(hwnd, title, titleLen + 1);
        } else {
            titleLen = 0;
        }
        if (titleLen > 0) {
            window->title = WIN_StringToUTF8(title);
        }
        if (title) {
            SDL_small_free(title, isstack);
        }
    }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

    SDL_PropertiesID props = SDL_GetWindowProperties(window);
    SDL_SetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, data->hwnd);
    SDL_SetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HDC_POINTER, data->hdc);
    SDL_SetPointerProperty(props, SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, data->hinstance);

    // All done!
    return true;
}

static void CleanupWindowData(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;

    if (data) {

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
        if (data->drop_target) {
            WIN_AcceptDragAndDrop(window, false);
        }
        if (data->ICMFileName) {
            SDL_free(data->ICMFileName);
        }
        if (data->keyboard_hook) {
            UnhookWindowsHookEx(data->keyboard_hook);
        }
        ReleaseDC(data->hwnd, data->hdc);
        RemoveProp(data->hwnd, TEXT("SDL_WindowData"));
#endif
        if (!(window->flags & SDL_WINDOW_EXTERNAL)) {
            DestroyWindow(data->hwnd);
            if (data->destroy_parent_with_window && data->parent) {
                DestroyWindow(data->parent);
            }
        } else {
            // Restore any original event handler...
            if (data->wndproc) {
#ifdef GWLP_WNDPROC
                SetWindowLongPtr(data->hwnd, GWLP_WNDPROC,
                                 (LONG_PTR)data->wndproc);
#else
                SetWindowLong(data->hwnd, GWL_WNDPROC,
                              (LONG_PTR)data->wndproc);
#endif
            }
        }
        SDL_free(data);
    }
    window->internal = NULL;
}

static void WIN_ConstrainPopup(SDL_Window *window, bool output_to_pending)
{
    // Possibly clamp popup windows to the output borders
    if (SDL_WINDOW_IS_POPUP(window)) {
        SDL_Window *w;
        SDL_DisplayID displayID;
        SDL_Rect rect;
        int abs_x = window->last_position_pending ? window->pending.x : window->floating.x;
        int abs_y = window->last_position_pending ? window->pending.y : window->floating.y;
        const int width = window->last_size_pending ? window->pending.w : window->floating.w;
        const int height = window->last_size_pending ? window->pending.h : window->floating.h;
        int offset_x = 0, offset_y = 0;

        if (window->constrain_popup) {
            // Calculate the total offset from the parents
            for (w = window->parent; SDL_WINDOW_IS_POPUP(w); w = w->parent) {
                offset_x += w->x;
                offset_y += w->y;
            }

            offset_x += w->x;
            offset_y += w->y;
            abs_x += offset_x;
            abs_y += offset_y;

            // Constrain the popup window to the display of the toplevel parent
            displayID = SDL_GetDisplayForWindow(w);
            SDL_GetDisplayBounds(displayID, &rect);
            if (abs_x + width > rect.x + rect.w) {
                abs_x -= (abs_x + width) - (rect.x + rect.w);
            }
            if (abs_y + height > rect.y + rect.h) {
                abs_y -= (abs_y + height) - (rect.y + rect.h);
            }
            abs_x = SDL_max(abs_x, rect.x);
            abs_y = SDL_max(abs_y, rect.y);
        }

        if (output_to_pending) {
            window->pending.x = abs_x - offset_x;
            window->pending.y = abs_y - offset_y;
            window->pending.w = width;
            window->pending.h = height;
        } else {
            window->floating.x = abs_x - offset_x;
            window->floating.y = abs_y - offset_y;
            window->floating.w = width;
            window->floating.h = height;
        }
    }
}

static void WIN_SetKeyboardFocus(SDL_Window *window, bool set_active_focus)
{
    SDL_Window *toplevel = window;

    // Find the topmost parent
    while (SDL_WINDOW_IS_POPUP(toplevel)) {
        toplevel = toplevel->parent;
    }

    toplevel->keyboard_focus = window;

    if (set_active_focus && !window->is_hiding && !window->is_destroying) {
        SDL_SetKeyboardFocus(window);
    }
}

bool WIN_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props)
{
    SDL_VideoData *videodata = _this->internal;
    HWND hwnd = (HWND)SDL_GetPointerProperty(create_props, SDL_PROP_WINDOW_CREATE_WIN32_HWND_POINTER, SDL_GetPointerProperty(create_props, "sdl2-compat.external_window", NULL));
    HWND parent = NULL;
    if (hwnd) {
        window->flags |= SDL_WINDOW_EXTERNAL;

        if (!SetupWindowData(_this, window, hwnd, parent)) {
            return false;
        }
    } else {
        DWORD style = STYLE_BASIC;
        DWORD styleEx = 0;
        int x, y;
        int w, h;

        if (window->flags & SDL_WINDOW_UTILITY) {
            parent = CreateWindow(SDL_Appname, TEXT(""), STYLE_BASIC, 0, 0, 32, 32, NULL, NULL, SDL_Instance, NULL);
        } else if (window->parent) {
            parent = window->parent->internal->hwnd;
        }

        style |= GetWindowStyle(window);
        styleEx |= GetWindowStyleEx(window);

        // Figure out what the window area will be
        WIN_ConstrainPopup(window, false);
        WIN_AdjustWindowRectWithStyle(window, style, styleEx, FALSE, &x, &y, &w, &h, SDL_WINDOWRECT_FLOATING);

        hwnd = CreateWindowEx(styleEx, SDL_Appname, TEXT(""), style,
                              x, y, w, h, parent, NULL, SDL_Instance, NULL);
        if (!hwnd) {
            return WIN_SetError("Couldn't create window");
        }

        WIN_UpdateDarkModeForHWND(hwnd);

        WIN_PumpEvents(_this);

        if (!SetupWindowData(_this, window, hwnd, parent)) {
            DestroyWindow(hwnd);
            if (parent) {
                DestroyWindow(parent);
            }
            return false;
        }

        // Inform Windows of the frame change so we can respond to WM_NCCALCSIZE
        SetWindowPos(hwnd, NULL, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_NOACTIVATE);

        if (window->flags & SDL_WINDOW_MINIMIZED) {
            /* TODO: We have to clear SDL_WINDOW_HIDDEN here to ensure the window flags match the window state. The
               window is already shown after this and windows with WS_MINIMIZE do not generate a WM_SHOWWINDOW. This
               means you can't currently create a window that is initially hidden and is minimized when shown.
            */
            window->flags &= ~SDL_WINDOW_HIDDEN;
            ShowWindow(hwnd, SW_SHOWMINNOACTIVE);
        }
    }

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    // FIXME: does not work on all hardware configurations with different renders (i.e. hybrid GPUs)
    if (window->flags & SDL_WINDOW_TRANSPARENT) {
        if (videodata->DwmEnableBlurBehindWindow) {
            /* The region indicates which part of the window will be blurred and rest will be transparent. This
               is because the alpha value of the window will be used for non-blurred areas
               We can use (-1, -1, 0, 0) boundary to make sure no pixels are being blurred
            */
            HRGN rgn = CreateRectRgn(-1, -1, 0, 0);
            DWM_BLURBEHIND bb;
            bb.flags = (DWM_BB_ENABLE | DWM_BB_BLURREGION);
            bb.enable = TRUE;
            bb.blur_region = rgn;
            bb.transition_on_maxed = FALSE;
            videodata->DwmEnableBlurBehindWindow(hwnd, &bb);
            DeleteObject(rgn);
        }
    }

    HWND share_hwnd = (HWND)SDL_GetPointerProperty(create_props, SDL_PROP_WINDOW_CREATE_WIN32_PIXEL_FORMAT_HWND_POINTER, NULL);
    if (share_hwnd) {
        HDC hdc = GetDC(share_hwnd);
        int pixel_format = GetPixelFormat(hdc);
        PIXELFORMATDESCRIPTOR pfd;

        SDL_zero(pfd);
        DescribePixelFormat(hdc, pixel_format, sizeof(pfd), &pfd);
        ReleaseDC(share_hwnd, hdc);

        if (!SetPixelFormat(window->internal->hdc, pixel_format, &pfd)) {
            WIN_DestroyWindow(_this, window);
            return WIN_SetError("SetPixelFormat()");
        }
    } else {
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
        if (!(window->flags & SDL_WINDOW_OPENGL)) {
            return true;
        }

        // The rest of this macro mess is for OpenGL or OpenGL ES windows
#ifdef SDL_VIDEO_OPENGL_ES2
        if ((_this->gl_config.profile_mask == SDL_GL_CONTEXT_PROFILE_ES ||
             SDL_GetHintBoolean(SDL_HINT_VIDEO_FORCE_EGL, false))
#ifdef SDL_VIDEO_OPENGL_WGL
             && (!_this->gl_data || WIN_GL_UseEGL(_this))
#endif // SDL_VIDEO_OPENGL_WGL
        ) {
#ifdef SDL_VIDEO_OPENGL_EGL
            if (!WIN_GLES_SetupWindow(_this, window)) {
                WIN_DestroyWindow(_this, window);
                return false;
            }
            return true;
#else
            return SDL_SetError("Could not create GLES window surface (EGL support not configured)");
#endif // SDL_VIDEO_OPENGL_EGL
        }
#endif // SDL_VIDEO_OPENGL_ES2

#ifdef SDL_VIDEO_OPENGL_WGL
        if (!WIN_GL_SetupWindow(_this, window)) {
            WIN_DestroyWindow(_this, window);
            return false;
        }
#else
        return SDL_SetError("Could not create GL window (WGL support not configured)");
#endif
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

    return true;
}

void WIN_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    HWND hwnd = window->internal->hwnd;
    LPTSTR title = WIN_UTF8ToString(window->title);
    SetWindowText(hwnd, title);
    SDL_free(title);
#endif
}

bool WIN_SetWindowIcon(SDL_VideoDevice *_this, SDL_Window *window, SDL_Surface *icon)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    HWND hwnd = window->internal->hwnd;
    HICON hicon = NULL;
    BYTE *icon_bmp;
    int icon_len, mask_len, row_len, y;
    BITMAPINFOHEADER *bmi;
    Uint8 *dst;
    bool isstack;
    bool result = true;

    // Create temporary buffer for ICONIMAGE structure
    SDL_COMPILE_TIME_ASSERT(WIN_SetWindowIcon_uses_BITMAPINFOHEADER_to_prepare_an_ICONIMAGE, sizeof(BITMAPINFOHEADER) == 40);
    mask_len = (icon->h * (icon->w + 7) / 8);
    icon_len = sizeof(BITMAPINFOHEADER) + icon->h * icon->w * sizeof(Uint32) + mask_len;
    icon_bmp = SDL_small_alloc(BYTE, icon_len, &isstack);
    if (!icon_bmp) {
        return false;
    }

    // Write the BITMAPINFO header
    bmi = (BITMAPINFOHEADER *)icon_bmp;
    bmi->biSize = SDL_Swap32LE(sizeof(BITMAPINFOHEADER));
    bmi->biWidth = SDL_Swap32LE(icon->w);
    bmi->biHeight = SDL_Swap32LE(icon->h * 2);
    bmi->biPlanes = SDL_Swap16LE(1);
    bmi->biBitCount = SDL_Swap16LE(32);
    bmi->biCompression = SDL_Swap32LE(BI_RGB);
    bmi->biSizeImage = SDL_Swap32LE(icon->h * icon->w * sizeof(Uint32));
    bmi->biXPelsPerMeter = SDL_Swap32LE(0);
    bmi->biYPelsPerMeter = SDL_Swap32LE(0);
    bmi->biClrUsed = SDL_Swap32LE(0);
    bmi->biClrImportant = SDL_Swap32LE(0);

    // Write the pixels upside down into the bitmap buffer
    SDL_assert(icon->format == SDL_PIXELFORMAT_ARGB8888);
    dst = &icon_bmp[sizeof(BITMAPINFOHEADER)];
    row_len = icon->w * sizeof(Uint32);
    y = icon->h;
    while (y--) {
        Uint8 *src = (Uint8 *)icon->pixels + y * icon->pitch;
        SDL_memcpy(dst, src, row_len);
        dst += row_len;
    }

    // Write the mask
    SDL_memset(icon_bmp + icon_len - mask_len, 0xFF, mask_len);

    hicon = CreateIconFromResource(icon_bmp, icon_len, TRUE, 0x00030000);

    SDL_small_free(icon_bmp, isstack);

    if (!hicon) {
        result = SDL_SetError("SetWindowIcon() failed, error %08X", (unsigned int)GetLastError());
    }

    // Set the icon for the window
    SendMessage(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)hicon);

    // Set the icon in the task manager (should we do this?)
    SendMessage(hwnd, WM_SETICON, ICON_BIG, (LPARAM)hicon);
    return result;
#else
    return SDL_Unsupported();
#endif
}

bool WIN_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window)
{
    /* HighDPI support: removed SWP_NOSIZE. If the move results in a DPI change, we need to allow
     * the window to resize (e.g. AdjustWindowRectExForDpi frame sizes are different).
     */
    if (!(window->flags & SDL_WINDOW_FULLSCREEN)) {
        if (!(window->flags & (SDL_WINDOW_MAXIMIZED | SDL_WINDOW_MINIMIZED))) {
            WIN_ConstrainPopup(window, true);
            return WIN_SetWindowPositionInternal(window,
                                                 window->internal->copybits_flag | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_NOSIZE |
                                                 SWP_NOACTIVATE, SDL_WINDOWRECT_PENDING);
        }
    } else {
        return SDL_UpdateFullscreenMode(window, SDL_FULLSCREEN_OP_ENTER, true);
    }

    return true;
}

void WIN_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window)
{
    if (!(window->flags & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_MAXIMIZED))) {
        WIN_SetWindowPositionInternal(window, window->internal->copybits_flag | SWP_NOMOVE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_NOACTIVATE, SDL_WINDOWRECT_PENDING);
    } else {
        // Can't resize the window
        window->last_size_pending = false;
    }
}

bool WIN_GetWindowBordersSize(SDL_VideoDevice *_this, SDL_Window *window, int *top, int *left, int *bottom, int *right)
{
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    HWND hwnd = window->internal->hwnd;
    RECT rcClient;

    /* rcClient stores the size of the inner window, while rcWindow stores the outer size relative to the top-left
     * screen position; so the top/left values of rcClient are always {0,0} and bottom/right are {height,width} */
    GetClientRect(hwnd, &rcClient);

    *top = rcClient.top;
    *left = rcClient.left;
    *bottom = rcClient.bottom;
    *right = rcClient.right;

    return true;
#else  // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    HWND hwnd = window->internal->hwnd;
    RECT rcClient, rcWindow;
    POINT ptDiff;

    /* rcClient stores the size of the inner window, while rcWindow stores the outer size relative to the top-left
     * screen position; so the top/left values of rcClient are always {0,0} and bottom/right are {height,width} */
    if (!GetClientRect(hwnd, &rcClient)) {
        return SDL_SetError("GetClientRect() failed, error %08X", (unsigned int)GetLastError());
    }

    if (!GetWindowRect(hwnd, &rcWindow)) {
        return SDL_SetError("GetWindowRect() failed, error %08X", (unsigned int)GetLastError());
    }

    /* convert the top/left values to make them relative to
     * the window; they will end up being slightly negative */
    ptDiff.y = rcWindow.top;
    ptDiff.x = rcWindow.left;

    if (!ScreenToClient(hwnd, &ptDiff)) {
        return SDL_SetError("ScreenToClient() failed, error %08X", (unsigned int)GetLastError());
    }

    rcWindow.top = ptDiff.y;
    rcWindow.left = ptDiff.x;

    /* convert the bottom/right values to make them relative to the window,
     * these will be slightly bigger than the inner width/height */
    ptDiff.y = rcWindow.bottom;
    ptDiff.x = rcWindow.right;

    if (!ScreenToClient(hwnd, &ptDiff)) {
        return SDL_SetError("ScreenToClient() failed, error %08X", (unsigned int)GetLastError());
    }

    rcWindow.bottom = ptDiff.y;
    rcWindow.right = ptDiff.x;

    /* Now that both the inner and outer rects use the same coordinate system we can subtract them to get the border size.
     * Keep in mind that the top/left coordinates of rcWindow are negative because the border lies slightly before {0,0},
     * so switch them around because SDL3 wants them in positive. */
    *top = rcClient.top - rcWindow.top;
    *left = rcClient.left - rcWindow.left;
    *bottom = rcWindow.bottom - rcClient.bottom;
    *right = rcWindow.right - rcClient.right;

    return true;
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
}

void WIN_GetWindowSizeInPixels(SDL_VideoDevice *_this, SDL_Window *window, int *w, int *h)
{
    const SDL_WindowData *data = window->internal;
    HWND hwnd = data->hwnd;
    RECT rect;

    if (GetClientRect(hwnd, &rect) && WIN_WindowRectValid(&rect)) {
        *w = rect.right;
        *h = rect.bottom;
    } else if (window->last_pixel_w && window->last_pixel_h) {
        *w = window->last_pixel_w;
        *h = window->last_pixel_h;
    } else {
        // Probably created minimized, use the restored size
        *w = window->floating.w;
        *h = window->floating.h;
    }
}

void WIN_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    DWORD style;
    HWND hwnd;

    bool bActivate = SDL_GetHintBoolean(SDL_HINT_WINDOW_ACTIVATE_WHEN_SHOWN, true);

    if (SDL_WINDOW_IS_POPUP(window)) {
        // Update our position in case our parent moved while we were hidden
        WIN_SetWindowPosition(_this, window);
    }

    hwnd = window->internal->hwnd;
    style = GetWindowLong(hwnd, GWL_EXSTYLE);
    if (style & WS_EX_NOACTIVATE) {
        bActivate = false;
    }
    if (bActivate) {
        ShowWindow(hwnd, SW_SHOW);
    } else {
        // Use SetWindowPos instead of ShowWindow to avoid activating the parent window if this is a child window
        SetWindowPos(hwnd, NULL, 0, 0, 0, 0, window->internal->copybits_flag | SWP_SHOWWINDOW | SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER);
    }

    if ((window->flags & SDL_WINDOW_POPUP_MENU) && !(window->flags & SDL_WINDOW_NOT_FOCUSABLE) && bActivate) {
        WIN_SetKeyboardFocus(window, true);
    }
    if (window->flags & SDL_WINDOW_MODAL) {
        WIN_SetWindowModal(_this, window, true);
    }
}

void WIN_HideWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    HWND hwnd = window->internal->hwnd;

    if (window->flags & SDL_WINDOW_MODAL) {
        WIN_SetWindowModal(_this, window, false);
    }

    ShowWindow(hwnd, SW_HIDE);

    // Transfer keyboard focus back to the parent from a grabbing popup.
    if ((window->flags & SDL_WINDOW_POPUP_MENU) && !(window->flags & SDL_WINDOW_NOT_FOCUSABLE)) {
        SDL_Window *new_focus;
        const bool set_focus = SDL_ShouldRelinquishPopupFocus(window, &new_focus);
        WIN_SetKeyboardFocus(new_focus, set_focus);
    }
}

void WIN_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    /* If desired, raise the window more forcefully.
     * Technique taken from http://stackoverflow.com/questions/916259/ .
     * Specifically, http://stackoverflow.com/a/34414846 .
     *
     * The issue is that Microsoft has gone through a lot of trouble to make it
     * nearly impossible to programmatically move a window to the foreground,
     * for "security" reasons. Apparently, the following song-and-dance gets
     * around their objections. */
    bool bForce = SDL_GetHintBoolean(SDL_HINT_FORCE_RAISEWINDOW, false);
    bool bActivate = SDL_GetHintBoolean(SDL_HINT_WINDOW_ACTIVATE_WHEN_RAISED, true);

    HWND hCurWnd = NULL;
    DWORD dwMyID = 0u;
    DWORD dwCurID = 0u;

    SDL_WindowData *data = window->internal;
    HWND hwnd = data->hwnd;
    if (bForce) {
        hCurWnd = GetForegroundWindow();
        dwMyID = GetCurrentThreadId();
        dwCurID = GetWindowThreadProcessId(hCurWnd, NULL);
        ShowWindow(hwnd, SW_RESTORE);
        AttachThreadInput(dwCurID, dwMyID, TRUE);
        SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE);
        if (!SDL_ShouldAllowTopmost() || !(window->flags & SDL_WINDOW_ALWAYS_ON_TOP)) {
            SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE);
        }
    }
    if (bActivate) {
        SetForegroundWindow(hwnd);
        if ((window->flags & SDL_WINDOW_POPUP_MENU) && !(window->flags & SDL_WINDOW_NOT_FOCUSABLE)) {
            WIN_SetKeyboardFocus(window, window->parent == SDL_GetKeyboardFocus());
        }
    } else {
        SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, data->copybits_flag | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_NOACTIVATE);
    }
    if (bForce) {
        AttachThreadInput(dwCurID, dwMyID, FALSE);
        SetFocus(hwnd);
        SetActiveWindow(hwnd);
    }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
}

void WIN_MaximizeWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;

    if (!(window->flags & SDL_WINDOW_FULLSCREEN)) {
        HWND hwnd = data->hwnd;
        data->expected_resize = true;
        ShowWindow(hwnd, SW_MAXIMIZE);
        data->expected_resize = false;

        /* Clamp the maximized window size to the max window size.
         * This is automatic if maximizing from the window controls.
         */
        if (window->max_w || window->max_h) {
            int fx, fy, fw, fh;

            window->windowed.w = window->max_w ? SDL_min(window->w, window->max_w) : window->windowed.w;
            window->windowed.h = window->max_h ? SDL_min(window->h, window->max_h) : window->windowed.h;
            WIN_AdjustWindowRect(window, &fx, &fy, &fw, &fh, SDL_WINDOWRECT_WINDOWED);

            data->expected_resize = true;
            SetWindowPos(hwnd, HWND_TOP, fx, fy, fw, fh, data->copybits_flag | SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOACTIVATE);
            data->expected_resize = false;
        }
    } else {
        data->windowed_mode_was_maximized = true;
    }
}

void WIN_MinimizeWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    HWND hwnd = window->internal->hwnd;
    ShowWindow(hwnd, SW_MINIMIZE);
}

void WIN_SetWindowBordered(SDL_VideoDevice *_this, SDL_Window *window, bool bordered)
{
    SDL_WindowData *data = window->internal;
    HWND hwnd = data->hwnd;
    DWORD style;

    style = GetWindowLong(hwnd, GWL_STYLE);
    style &= ~STYLE_MASK;
    style |= GetWindowStyle(window);

    data->in_border_change = true;
    SetWindowLong(hwnd, GWL_STYLE, style);
    WIN_SetWindowPositionInternal(window, data->copybits_flag | SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_NOACTIVATE, SDL_WINDOWRECT_CURRENT);
    data->in_border_change = false;
}

void WIN_SetWindowResizable(SDL_VideoDevice *_this, SDL_Window *window, bool resizable)
{
    SDL_WindowData *data = window->internal;
    HWND hwnd = data->hwnd;
    DWORD style;

    style = GetWindowLong(hwnd, GWL_STYLE);
    style &= ~STYLE_MASK;
    style |= GetWindowStyle(window);

    SetWindowLong(hwnd, GWL_STYLE, style);
}

void WIN_SetWindowAlwaysOnTop(SDL_VideoDevice *_this, SDL_Window *window, bool on_top)
{
    WIN_SetWindowPositionInternal(window, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOOWNERZORDER, SDL_WINDOWRECT_CURRENT);
}

void WIN_RestoreWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;
    if (!(window->flags & SDL_WINDOW_FULLSCREEN)) {
        HWND hwnd = data->hwnd;
        data->expected_resize = true;
        ShowWindow(hwnd, SW_RESTORE);
        data->expected_resize = false;
    } else {
        data->windowed_mode_was_maximized = false;
    }
}

static void WIN_UpdateCornerRoundingForHWND(SDL_VideoDevice *_this, HWND hwnd, DWM_WINDOW_CORNER_PREFERENCE cornerPref)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    SDL_VideoData *videodata = _this->internal;
    if (videodata->DwmSetWindowAttribute) {
        videodata->DwmSetWindowAttribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, &cornerPref, sizeof(cornerPref));
    }
#endif
}

static void WIN_UpdateBorderColorForHWND(SDL_VideoDevice *_this, HWND hwnd, COLORREF colorRef)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    SDL_VideoData *videodata = _this->internal;
    if (videodata->DwmSetWindowAttribute) {
        videodata->DwmSetWindowAttribute(hwnd, DWMWA_BORDER_COLOR, &colorRef, sizeof(colorRef));
    }
#endif
}

/**
 * Reconfigures the window to fill the given display, if fullscreen is true, otherwise restores the window.
 */
SDL_FullscreenResult WIN_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *display, SDL_FullscreenOp fullscreen)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    SDL_DisplayData *displaydata = display->internal;
    SDL_WindowData *data = window->internal;
    HWND hwnd = data ? data->hwnd : NULL;
    MONITORINFO minfo;
    DWORD style, styleEx;
    HWND top;
    int x, y;
    int w, h;
    bool enterMaximized = false;

#ifdef HIGHDPI_DEBUG
    SDL_Log("WIN_SetWindowFullscreen: %d", (int)fullscreen);
#endif

    /* Early out if already not in fullscreen, or the styling on
     * external windows may end up being overridden.
     */
    if (!(window->flags & SDL_WINDOW_FULLSCREEN) && !fullscreen) {
        return SDL_FULLSCREEN_SUCCEEDED;
    }

    if (SDL_ShouldAllowTopmost() && (window->flags & SDL_WINDOW_ALWAYS_ON_TOP)) {
        top = HWND_TOPMOST;
    } else {
        top = HWND_NOTOPMOST;
    }

    /* Use GetMonitorInfo instead of WIN_GetDisplayBounds because we want the
       monitor bounds in Windows coordinates (pixels) rather than SDL coordinates (points). */
    SDL_zero(minfo);
    minfo.cbSize = sizeof(MONITORINFO);
    if (!GetMonitorInfo(displaydata->MonitorHandle, &minfo)) {
        SDL_SetError("GetMonitorInfo failed");
        return SDL_FULLSCREEN_FAILED;
    }

    SDL_SendWindowEvent(window, fullscreen ? SDL_EVENT_WINDOW_ENTER_FULLSCREEN : SDL_EVENT_WINDOW_LEAVE_FULLSCREEN, 0, 0);
    style = GetWindowLong(hwnd, GWL_STYLE);
    style &= ~STYLE_MASK;
    style |= GetWindowStyle(window);
    styleEx = GetWindowLong(hwnd, GWL_EXSTYLE);

    if (fullscreen) {
        x = minfo.rcMonitor.left;
        y = minfo.rcMonitor.top;
        w = minfo.rcMonitor.right - minfo.rcMonitor.left;
        h = minfo.rcMonitor.bottom - minfo.rcMonitor.top;

        /* Unset the maximized flag.  This fixes
           https://bugzilla.libsdl.org/show_bug.cgi?id=3215
        */
        if (style & WS_MAXIMIZE) {
            data->windowed_mode_was_maximized = true;
            style &= ~WS_MAXIMIZE;
        }

        // Disable corner rounding & border color (Windows 11+) so the window fills the full screen
        WIN_UpdateCornerRoundingForHWND(_this, hwnd, DWMWCP_DONOTROUND);
        WIN_UpdateBorderColorForHWND(_this, hwnd, DWMWA_COLOR_NONE);
    } else {
        BOOL menu;

        WIN_UpdateCornerRoundingForHWND(_this, hwnd, DWMWCP_DEFAULT);
        WIN_UpdateBorderColorForHWND(_this, hwnd, DWMWA_COLOR_DEFAULT);

        /* Restore window-maximization state, as applicable.
           Special care is taken to *not* do this if and when we're
           alt-tab'ing away (to some other window; as indicated by
           in_window_deactivation), otherwise
           https://bugzilla.libsdl.org/show_bug.cgi?id=3215 can reproduce!
        */
        if (data->windowed_mode_was_maximized && !data->in_window_deactivation) {
            enterMaximized = true;
            data->disable_move_size_events = true;
        }

        menu = (style & WS_CHILDWINDOW) ? FALSE : (GetMenu(hwnd) != NULL);
        WIN_AdjustWindowRectWithStyle(window, style, styleEx, menu,
                                      &x, &y,
                                      &w, &h,
                                      SDL_WINDOWRECT_FLOATING);
        data->windowed_mode_was_maximized = false;

        /* A window may have been maximized by dragging it to the top of another display, in which case the floating
         * position may be out-of-date. If the window is being restored to maximized, and the maximized and floating
         * position are on different displays, try to center the window on the maximized display for restoration, which
         * mimics native Windows behavior.
         */
        if (enterMaximized) {
            const SDL_Point windowed_point = { window->windowed.x, window->windowed.y };
            const SDL_Point floating_point = { window->floating.x, window->floating.y };
            const SDL_DisplayID floating_display = SDL_GetDisplayForPoint(&floating_point);
            const SDL_DisplayID windowed_display = SDL_GetDisplayForPoint(&windowed_point);

            if (floating_display != windowed_display) {
                SDL_Rect bounds;

                SDL_zero(bounds);
                SDL_GetDisplayUsableBounds(windowed_display, &bounds);
                if (w < bounds.w) {
                    x = bounds.x + (bounds.w - w) / 2;
                } else {
                    x = bounds.x;
                }
                if (h < bounds.h) {
                    y = bounds.y + (bounds.h - h) / 2;
                } else {
                    y = bounds.y;
                }
            }
        }
    }

    /* Always reset the window to the base floating size before possibly re-applying the maximized state,
     * otherwise, the base floating size can seemingly be lost in some cases.
     */
    SetWindowLong(hwnd, GWL_STYLE, style);
    data->expected_resize = true;
    SetWindowPos(hwnd, top, x, y, w, h, data->copybits_flag | SWP_NOACTIVATE);
    data->expected_resize = false;
    data->disable_move_size_events = false;

    if (enterMaximized) {
        WIN_MaximizeWindow(_this, window);
    }

#ifdef HIGHDPI_DEBUG
    SDL_Log("WIN_SetWindowFullscreen: %d finished. Set window to %d,%d, %dx%d", (int)fullscreen, x, y, w, h);
#endif

#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    return SDL_FULLSCREEN_SUCCEEDED;
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
void WIN_UpdateWindowICCProfile(SDL_Window *window, bool send_event)
{
    SDL_WindowData *data = window->internal;
    SDL_DisplayData *displaydata = SDL_GetDisplayDriverDataForWindow(window);

    if (displaydata) {
        HDC hdc = CreateDCW(displaydata->DeviceName, NULL, NULL, NULL);
        if (hdc) {
            WCHAR fileName[MAX_PATH];
            DWORD fileNameSize = SDL_arraysize(fileName);
            if (GetICMProfileW(hdc, &fileNameSize, fileName)) {
                // fileNameSize includes '\0' on return
                if (!data->ICMFileName ||
                    SDL_wcscmp(data->ICMFileName, fileName) != 0) {
                    if (data->ICMFileName) {
                        SDL_free(data->ICMFileName);
                    }
                    data->ICMFileName = SDL_wcsdup(fileName);
                    if (send_event) {
                        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_ICCPROF_CHANGED, 0, 0);
                    }
                }
            }
            DeleteDC(hdc);
        }
    }
}

void *WIN_GetWindowICCProfile(SDL_VideoDevice *_this, SDL_Window *window, size_t *size)
{
    SDL_WindowData *data = window->internal;
    char *filename_utf8;
    void *iccProfileData = NULL;

    filename_utf8 = WIN_StringToUTF8(data->ICMFileName);
    if (filename_utf8) {
        iccProfileData = SDL_LoadFile(filename_utf8, size);
        if (!iccProfileData) {
            SDL_SetError("Could not open ICC profile");
        }
        SDL_free(filename_utf8);
    }
    return iccProfileData;
}

static void WIN_GrabKeyboard(SDL_Window *window)
{
    SDL_WindowData *data = window->internal;
    HMODULE module;

    if (data->keyboard_hook) {
        return;
    }

    /* SetWindowsHookEx() needs to know which module contains the hook we
       want to install. This is complicated by the fact that SDL can be
       linked statically or dynamically. Fortunately XP and later provide
       this nice API that will go through the loaded modules and find the
       one containing our code.
    */
    if (!GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                           (LPTSTR)WIN_KeyboardHookProc,
                           &module)) {
        return;
    }

    // Capture a snapshot of the current keyboard state before the hook
    if (!GetKeyboardState(data->videodata->pre_hook_key_state)) {
        return;
    }

    /* To grab the keyboard, we have to install a low-level keyboard hook to
       intercept keys that would normally be captured by the OS. Intercepting
       all key events on the system is rather invasive, but it's what Microsoft
       actually documents that you do to capture these.
    */
    data->keyboard_hook = SetWindowsHookEx(WH_KEYBOARD_LL, WIN_KeyboardHookProc, module, 0);
}

void WIN_UngrabKeyboard(SDL_Window *window)
{
    SDL_WindowData *data = window->internal;

    if (data->keyboard_hook) {
        UnhookWindowsHookEx(data->keyboard_hook);
        data->keyboard_hook = NULL;
    }
}

bool WIN_SetWindowMouseRect(SDL_VideoDevice *_this, SDL_Window *window)
{
    WIN_UpdateClipCursor(window);
    return true;
}

bool WIN_SetWindowMouseGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed)
{
    WIN_UpdateClipCursor(window);
    return true;
}

bool WIN_SetWindowKeyboardGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed)
{
    if (grabbed) {
        WIN_GrabKeyboard(window);
    } else {
        WIN_UngrabKeyboard(window);
    }

    return true;
}
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

void WIN_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    CleanupWindowData(_this, window);
}

/*
 * Creates a HelperWindow used for DirectInput.
 */
bool SDL_HelperWindowCreate(void)
{
    HINSTANCE hInstance = GetModuleHandle(NULL);
    WNDCLASS wce;

    // Make sure window isn't created twice.
    if (SDL_HelperWindow != NULL) {
        return true;
    }

    // Create the class.
    SDL_zero(wce);
    wce.lpfnWndProc = DefWindowProc;
    wce.lpszClassName = SDL_HelperWindowClassName;
    wce.hInstance = hInstance;

    // Register the class.
    SDL_HelperWindowClass = RegisterClass(&wce);
    if (SDL_HelperWindowClass == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        return WIN_SetError("Unable to create Helper Window Class");
    }

    // Create the window.
    SDL_HelperWindow = CreateWindowEx(0, SDL_HelperWindowClassName,
                                      SDL_HelperWindowName,
                                      WS_OVERLAPPED, CW_USEDEFAULT,
                                      CW_USEDEFAULT, CW_USEDEFAULT,
                                      CW_USEDEFAULT, HWND_MESSAGE, NULL,
                                      hInstance, NULL);
    if (!SDL_HelperWindow) {
        UnregisterClass(SDL_HelperWindowClassName, hInstance);
        return WIN_SetError("Unable to create Helper Window");
    }

    return true;
}

/*
 * Destroys the HelperWindow previously created with SDL_HelperWindowCreate.
 */
void SDL_HelperWindowDestroy(void)
{
    HINSTANCE hInstance = GetModuleHandle(NULL);

    // Destroy the window.
    if (SDL_HelperWindow != NULL) {
        if (DestroyWindow(SDL_HelperWindow) == 0) {
            WIN_SetError("Unable to destroy Helper Window");
            return;
        }
        SDL_HelperWindow = NULL;
    }

    // Unregister the class.
    if (SDL_HelperWindowClass != 0) {
        if ((UnregisterClass(SDL_HelperWindowClassName, hInstance)) == 0) {
            WIN_SetError("Unable to destroy Helper Window Class");
            return;
        }
        SDL_HelperWindowClass = 0;
    }
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
void WIN_OnWindowEnter(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;

    if (!data || !data->hwnd) {
        // The window wasn't fully initialized
        return;
    }

    if (window->flags & SDL_WINDOW_ALWAYS_ON_TOP) {
        WIN_SetWindowPositionInternal(window, data->copybits_flag | SWP_NOSIZE | SWP_NOACTIVATE, SDL_WINDOWRECT_CURRENT);
    }
}

static BOOL GetClientScreenRect(HWND hwnd, RECT *rect)
{
    return GetClientRect(hwnd, rect) &&             // RECT( left , top , right , bottom )
           ClientToScreen(hwnd, (LPPOINT)rect) &&   // POINT( left , top )
           ClientToScreen(hwnd, (LPPOINT)rect + 1); // POINT( right , bottom )
}

void WIN_UnclipCursorForWindow(SDL_Window *window) {
    SDL_WindowData *data = window->internal;
    RECT rect;
    if (GetClipCursor(&rect) && SDL_memcmp(&rect, &data->cursor_clipped_rect, sizeof(rect)) == 0) {
        ClipCursor(NULL);
        SDL_zero(data->cursor_clipped_rect);
    }
}

void WIN_UpdateClipCursor(SDL_Window *window)
{
    SDL_WindowData *data = window->internal;
    if (data->in_title_click || data->focus_click_pending || data->postpone_clipcursor) {
        return;
    }

    SDL_Rect mouse_rect = window->mouse_rect;
    bool win_mouse_rect = (mouse_rect.w > 0 && mouse_rect.h > 0);
    bool win_have_focus = (window->flags & SDL_WINDOW_INPUT_FOCUS);
    bool win_is_grabbed = (window->flags & SDL_WINDOW_MOUSE_GRABBED);
    bool win_in_relmode = (window->flags & SDL_WINDOW_MOUSE_RELATIVE_MODE);
    bool cursor_confine = win_in_relmode || win_is_grabbed || win_mouse_rect;

    // This is verbatim translation of the old logic,
    // but I don't quite get what it's trying to do.
    // A clean-room implementation according to MSDN
    // documentation of GetClipCursor is provided in
    // a commented-out block below.
    if (!win_have_focus || !cursor_confine) {
        SDL_VideoDevice *videodevice = SDL_GetVideoDevice();
        RECT current;
        if (!GetClipCursor(&current)) {
            return;
        }
        if (videodevice && (
            current.left != videodevice->desktop_bounds.x ||
            current.top  != videodevice->desktop_bounds.y
        )) {
            POINT first, second;
            first.x  = current.left;
            first.y  = current.top;
            second.x = current.right  - 1;
            second.y = current.bottom - 1;
            if (!PtInRect(&data->cursor_clipped_rect, first) ||
                !PtInRect(&data->cursor_clipped_rect, second)) {
                return;
            }
        }
        ClipCursor(NULL);
        SDL_zero(data->cursor_clipped_rect);
        return;
    }

    // if (!win_have_focus || !cursor_confine) {
    //     RECT current;
    //     SDL_VideoDevice *videodevice = SDL_GetVideoDevice();
    //     if (GetClipCursor(&current) && (!videodevice ||
    //         current.left   != videodevice->desktop_bounds.x ||
    //         current.top    != videodevice->desktop_bounds.y ||
    //         current.right  != videodevice->desktop_bounds.x + videodevice->desktop_bounds.w ||
    //         current.bottom != videodevice->desktop_bounds.y + videodevice->desktop_bounds.h )) {
    //         ClipCursor(NULL);
    //         SDL_zero(data->cursor_clipped_rect);
    //     }
    //     return;
    // }

    SDL_Mouse *mouse = SDL_GetMouse();
    bool lock_to_ctr = (mouse->relative_mode && mouse->relative_mode_center);

    RECT client;
    if (!GetClientScreenRect(data->hwnd, &client)) {
        return;
    }

    RECT target = client;
    if (lock_to_ctr) {
        LONG cx = (client.left + client.right ) / 2;
        LONG cy = (client.top  + client.bottom) / 2;
        target = data->cursor_ctrlock_rect;
        target.left   += cx;
        target.right  += cx;
        target.top    += cy;
        target.bottom += cy;
    } else if (win_mouse_rect) {
        RECT custom, overlap;
        custom.left   = client.left + mouse_rect.x;
        custom.top    = client.top  + mouse_rect.y;
        custom.right  = client.left + mouse_rect.x + mouse_rect.w;
        custom.bottom = client.top  + mouse_rect.y + mouse_rect.h;
        if (IntersectRect(&overlap, &client, &custom)) {
            target = overlap;
        } else if (!win_is_grabbed) {
            WIN_UnclipCursorForWindow(window);
            return;
        }
    }

    if (GetClipCursor(&client) &&
        0 != SDL_memcmp(&target, &client, sizeof(client)) &&
        ClipCursor(&target)) {
        data->cursor_clipped_rect = target; // ClipCursor may fail if rect beyond screen
    }
}

bool WIN_SetWindowHitTest(SDL_Window *window, bool enabled)
{
    return true; // just succeed, the real work is done elsewhere.
}
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

bool WIN_SetWindowOpacity(SDL_VideoDevice *_this, SDL_Window *window, float opacity)
{
#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)
    return false;
#else
    const SDL_WindowData *data = window->internal;
    HWND hwnd = data->hwnd;
    const LONG style = GetWindowLong(hwnd, GWL_EXSTYLE);

    SDL_assert(style != 0);

    if (opacity == 1.0f) {
        // want it fully opaque, just mark it unlayered if necessary.
        if (style & WS_EX_LAYERED) {
            if (SetWindowLong(hwnd, GWL_EXSTYLE, style & ~WS_EX_LAYERED) == 0) {
                return WIN_SetError("SetWindowLong()");
            }
        }
    } else {
        const BYTE alpha = (BYTE)((int)(opacity * 255.0f));
        // want it transparent, mark it layered if necessary.
        if (!(style & WS_EX_LAYERED)) {
            if (SetWindowLong(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED) == 0) {
                return WIN_SetError("SetWindowLong()");
            }
        }

        if (SetLayeredWindowAttributes(hwnd, 0, alpha, LWA_ALPHA) == 0) {
            return WIN_SetError("SetLayeredWindowAttributes()");
        }
    }

    return true;
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

static const char *SDLGetClipboardFormatName(UINT cf, char *text, int len)
{
    switch (cf) {
    case CF_TEXT:
        return "CF_TEXT";
    case CF_BITMAP:
        return "CF_BITMAP";
    case CF_METAFILEPICT:
        return "CF_METAFILEPICT";
    case CF_SYLK:
        return "CF_SYLK";
    case CF_DIF:
        return "CF_DIF";
    case CF_TIFF:
        return "CF_TIFF";
    case CF_OEMTEXT:
        return "CF_OEMTEXT";
    case CF_DIB:
        return "CF_DIB";
    case CF_PALETTE:
        return "CF_PALETTE";
    case CF_PENDATA:
        return "CF_PENDATA";
    case CF_RIFF:
        return "CF_RIFF";
    case CF_WAVE:
        return "CF_WAVE";
    case CF_UNICODETEXT:
        return "CF_UNICODETEXT";
    case CF_ENHMETAFILE:
        return "CF_ENHMETAFILE";
    case CF_HDROP:
        return "CF_HDROP";
    case CF_LOCALE:
        return "CF_LOCALE";
    case CF_DIBV5:
        return "CF_DIBV5";
    case CF_OWNERDISPLAY:
        return "CF_OWNERDISPLAY";
    case CF_DSPTEXT:
        return "CF_DSPTEXT";
    case CF_DSPBITMAP:
        return "CF_DSPBITMAP";
    case CF_DSPMETAFILEPICT:
        return "CF_DSPMETAFILEPICT";
    case CF_DSPENHMETAFILE:
        return "CF_DSPENHMETAFILE";
    default:
        if (GetClipboardFormatNameA(cf, text, len)) {
            return text;
        } else {
            return NULL;
        }
    }
}

static STDMETHODIMP_(ULONG) SDLDropTarget_AddRef(SDLDropTarget *target)
{
    return ++target->refcount;
}

static STDMETHODIMP_(ULONG) SDLDropTarget_Release(SDLDropTarget *target)
{
    --target->refcount;
    if (target->refcount == 0) {
        SDL_free(target);
        return 0;
    }
    return target->refcount;
}

static STDMETHODIMP SDLDropTarget_QueryInterface(SDLDropTarget *target, REFIID riid, PVOID *ppv)
{
    if (ppv == NULL) {
        return E_INVALIDARG;
    }

    *ppv = NULL;
    if (WIN_IsEqualIID(riid, &IID_IUnknown) ||
        WIN_IsEqualIID(riid, &IID_IDropTarget)) {
        *ppv = (void *)target;
    }
    if (*ppv) {
        SDLDropTarget_AddRef(target);
        return S_OK;
    }
    return E_NOINTERFACE;
}

static STDMETHODIMP SDLDropTarget_DragEnter(SDLDropTarget *target,
                                            IDataObject *pDataObject, DWORD grfKeyState,
                                            POINTL pt, DWORD *pdwEffect)
{
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In DragEnter at %ld, %ld", pt.x, pt.y);
    *pdwEffect = DROPEFFECT_COPY;
    POINT pnt = { pt.x, pt.y };
    if (ScreenToClient(target->hwnd, &pnt)) {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In DragEnter at %ld, %ld => window %u at %ld, %ld", pt.x, pt.y, target->window->id, pnt.x, pnt.y);
        SDL_SendDropPosition(target->window, pnt.x, pnt.y);
    } else {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In DragEnter at %ld, %ld => nil, nil", pt.x, pt.y);
    }
    return S_OK;
}

static STDMETHODIMP SDLDropTarget_DragOver(SDLDropTarget *target,
                                           DWORD grfKeyState,
                                           POINTL pt, DWORD *pdwEffect)
{
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In DragOver at %ld, %ld", pt.x, pt.y);
    *pdwEffect = DROPEFFECT_COPY;
    POINT pnt = { pt.x, pt.y };
    if (ScreenToClient(target->hwnd, &pnt)) {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In DragOver at %ld, %ld => window %u at %ld, %ld", pt.x, pt.y, target->window->id, pnt.x, pnt.y);
        SDL_SendDropPosition(target->window, pnt.x, pnt.y);
    } else {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In DragOver at %ld, %ld => nil, nil", pt.x, pt.y);
    }
    return S_OK;
}

static STDMETHODIMP SDLDropTarget_DragLeave(SDLDropTarget *target)
{
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In DragLeave");
    SDL_SendDropComplete(target->window);
    return S_OK;
}

static STDMETHODIMP SDLDropTarget_Drop(SDLDropTarget *target,
                                       IDataObject *pDataObject, DWORD grfKeyState,
                                       POINTL pt, DWORD *pdwEffect)
{
    *pdwEffect = DROPEFFECT_COPY;
    POINT pnt = { pt.x, pt.y };
    if (ScreenToClient(target->hwnd, &pnt)) {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In Drop at %ld, %ld => window %u at %ld, %ld", pt.x, pt.y, target->window->id, pnt.x, pnt.y);
        SDL_SendDropPosition(target->window, pnt.x, pnt.y);
    } else {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In Drop at %ld, %ld => nil, nil", pt.x, pt.y);
    }

    {
        IEnumFORMATETC *pEnumFormatEtc;
        HRESULT hres;
        hres = pDataObject->lpVtbl->EnumFormatEtc(pDataObject, DATADIR_GET, &pEnumFormatEtc);
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In Drop for EnumFormatEtc, HRESULT is %08lx", hres);
        if (hres == S_OK) {
            FORMATETC fetc;
            while (pEnumFormatEtc->lpVtbl->Next(pEnumFormatEtc, 1, &fetc, NULL) == S_OK) {
                char name[257] = { 0 };
                const char *cfnm = SDLGetClipboardFormatName(fetc.cfFormat, name, 256);
                if (cfnm) {
                    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                                 ". In Drop, Supported format is %08x, '%s'", fetc.cfFormat, cfnm);
                } else {
                    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                                 ". In Drop, Supported format is %08x, Predefined", fetc.cfFormat);
                }
            }
        }
    }

    {
        FORMATETC fetc;
        fetc.cfFormat = target->format_file;
        fetc.ptd = NULL;
        fetc.dwAspect = DVASPECT_CONTENT;
        fetc.lindex = -1;
        fetc.tymed = TYMED_HGLOBAL;
        const char *format_mime = "text/uri-list";
        if (SUCCEEDED(pDataObject->lpVtbl->QueryGetData(pDataObject, &fetc))) {
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Drop File for QueryGetData, format %08x '%s', success",
                         fetc.cfFormat, format_mime);
            STGMEDIUM med;
            HRESULT hres = pDataObject->lpVtbl->GetData(pDataObject, &fetc, &med);
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Drop File for      GetData, format %08x '%s', HRESULT is %08lx",
                         fetc.cfFormat, format_mime, hres);
            if (SUCCEEDED(hres)) {
                const size_t bsize = GlobalSize(med.hGlobal);
                const void *buffer = (void *)GlobalLock(med.hGlobal);
                SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                             ". In Drop File for   GlobalLock, format %08x '%s', memory (%lu) %p",
                             fetc.cfFormat, format_mime, (unsigned long)bsize, buffer);
                if (buffer) {
                    char *text = (char *)SDL_malloc(bsize + sizeof(Uint32));
                    SDL_memcpy((Uint8 *)text, buffer, bsize);
                    SDL_memset((Uint8 *)text + bsize, 0, sizeof(Uint32));
                    char *saveptr = NULL;
                    char *token = SDL_strtok_r(text, "\r\n", &saveptr);
                    while (token != NULL) {
                        if (SDL_URIToLocal(token, token) >= 0) {
                            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                                         ". In Drop File, file (%lu of %lu) '%s'",
                                         (unsigned long)SDL_strlen(token), (unsigned long)bsize, token);
                            SDL_SendDropFile(target->window, NULL, token);
                        }
                        token = SDL_strtok_r(NULL, "\r\n", &saveptr);
                    }
                    SDL_free(text);
                }
                GlobalUnlock(med.hGlobal);
                ReleaseStgMedium(&med);
                SDL_SendDropComplete(target->window);
                return S_OK;
            }
        }
    }

    {
        FORMATETC fetc;
        fetc.cfFormat = target->format_text;
        fetc.ptd = NULL;
        fetc.dwAspect = DVASPECT_CONTENT;
        fetc.lindex = -1;
        fetc.tymed = TYMED_HGLOBAL;
        const char *format_mime = "text/plain;charset=utf-8";
        if (SUCCEEDED(pDataObject->lpVtbl->QueryGetData(pDataObject, &fetc))) {
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Drop Text for QueryGetData, format %08x '%s', success",
                         fetc.cfFormat, format_mime);
            STGMEDIUM med;
            HRESULT hres = pDataObject->lpVtbl->GetData(pDataObject, &fetc, &med);
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Drop Text for      GetData, format %08x '%s', HRESULT is %08lx",
                         fetc.cfFormat, format_mime, hres);
            if (SUCCEEDED(hres)) {
                const size_t bsize = GlobalSize(med.hGlobal);
                const void *buffer = (void *)GlobalLock(med.hGlobal);
                SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                             ". In Drop Text for   GlobalLock, format %08x '%s', memory (%lu) %p",
                             fetc.cfFormat, format_mime, (unsigned long)bsize, buffer);
                if (buffer) {
                    char *text = (char *)SDL_malloc(bsize + sizeof(Uint32));
                    SDL_memcpy((Uint8 *)text, buffer, bsize);
                    SDL_memset((Uint8 *)text + bsize, 0, sizeof(Uint32));
                    char *saveptr = NULL;
                    char *token = SDL_strtok_r(text, "\r\n", &saveptr);
                    while (token != NULL) {
                        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                                     ". In Drop Text, text (%lu of %lu) '%s'",
                                     (unsigned long)SDL_strlen(token), (unsigned long)bsize, token);
                        SDL_SendDropText(target->window, (char *)token);
                        token = SDL_strtok_r(NULL, "\r\n", &saveptr);
                    }
                    SDL_free(text);
                }
                GlobalUnlock(med.hGlobal);
                ReleaseStgMedium(&med);
                SDL_SendDropComplete(target->window);
                return S_OK;
            }
        }
    }

    {
        FORMATETC fetc;
        fetc.cfFormat = CF_UNICODETEXT;
        fetc.ptd = NULL;
        fetc.dwAspect = DVASPECT_CONTENT;
        fetc.lindex = -1;
        fetc.tymed = TYMED_HGLOBAL;
        const char *format_mime = "CF_UNICODETEXT";
        if (SUCCEEDED(pDataObject->lpVtbl->QueryGetData(pDataObject, &fetc))) {
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Drop Text for QueryGetData, format %08x '%s', success",
                         fetc.cfFormat, format_mime);
            STGMEDIUM med;
            HRESULT hres = pDataObject->lpVtbl->GetData(pDataObject, &fetc, &med);
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Drop Text for      GetData, format %08x '%s', HRESULT is %08lx",
                         fetc.cfFormat, format_mime, hres);
            if (SUCCEEDED(hres)) {
                const size_t bsize = GlobalSize(med.hGlobal);
                const void *buffer = (void *)GlobalLock(med.hGlobal);
                SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                             ". In Drop Text for   GlobalLock, format %08x '%s', memory (%lu) %p",
                             fetc.cfFormat, format_mime, (unsigned long)bsize, buffer);
                if (buffer) {
                    buffer = WIN_StringToUTF8((const wchar_t *)buffer);
                    if (buffer) {
                        const size_t lbuffer = SDL_strlen((const char *)buffer);
                        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                                     ". In Drop Text for StringToUTF8, format %08x '%s', memory (%lu) %p",
                                     fetc.cfFormat, format_mime, (unsigned long)lbuffer, buffer);
                        char *text = (char *)SDL_malloc(lbuffer + sizeof(Uint32));
                        SDL_memcpy((Uint8 *)text, buffer, lbuffer);
                        SDL_memset((Uint8 *)text + lbuffer, 0, sizeof(Uint32));
                        char *saveptr = NULL;
                        char *token = SDL_strtok_r(text, "\r\n", &saveptr);
                        while (token != NULL) {
                            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                                         ". In Drop Text, text (%lu of %lu) '%s'",
                                         (unsigned long)SDL_strlen(token), (unsigned long)lbuffer, token);
                            SDL_SendDropText(target->window, (char *)token);
                            token = SDL_strtok_r(NULL, "\r\n", &saveptr);
                        }
                        SDL_free(text);
                        SDL_free((void *)buffer);
                    }
                }
                GlobalUnlock(med.hGlobal);
                ReleaseStgMedium(&med);
                SDL_SendDropComplete(target->window);
                return S_OK;
            }
        }
    }

    {
        FORMATETC fetc;
        fetc.cfFormat = CF_TEXT;
        fetc.ptd = NULL;
        fetc.dwAspect = DVASPECT_CONTENT;
        fetc.lindex = -1;
        fetc.tymed = TYMED_HGLOBAL;
        const char *format_mime = "CF_TEXT";
        if (SUCCEEDED(pDataObject->lpVtbl->QueryGetData(pDataObject, &fetc))) {
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Drop Text for QueryGetData, format %08x '%s', success",
                         fetc.cfFormat, format_mime);
            STGMEDIUM med;
            HRESULT hres = pDataObject->lpVtbl->GetData(pDataObject, &fetc, &med);
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Drop Text for      GetData, format %08x '%s', HRESULT is %08lx",
                         fetc.cfFormat, format_mime, hres);
            if (SUCCEEDED(hres)) {
                const size_t bsize = GlobalSize(med.hGlobal);
                const void *buffer = (void *)GlobalLock(med.hGlobal);
                SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                             ". In Drop Text for   GlobalLock, format %08x '%s', memory (%lu) %p",
                             fetc.cfFormat, format_mime, (unsigned long)bsize, buffer);
                if (buffer) {
                    char *text = (char *)SDL_malloc(bsize + sizeof(Uint32));
                    SDL_memcpy((Uint8 *)text, buffer, bsize);
                    SDL_memset((Uint8 *)text + bsize, 0, sizeof(Uint32));
                    char *saveptr = NULL;
                    char *token = SDL_strtok_r(text, "\r\n", &saveptr);
                    while (token != NULL) {
                        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                                     ". In Drop Text, text (%lu of %lu) '%s'",
                                     (unsigned long)SDL_strlen(token), (unsigned long)bsize, token);
                        SDL_SendDropText(target->window, (char *)token);
                        token = SDL_strtok_r(NULL, "\r\n", &saveptr);
                    }
                    SDL_free(text);
                }
                GlobalUnlock(med.hGlobal);
                ReleaseStgMedium(&med);
                SDL_SendDropComplete(target->window);
                return S_OK;
            }
        }
    }

    {
        FORMATETC fetc;
        fetc.cfFormat = CF_HDROP;
        fetc.ptd = NULL;
        fetc.dwAspect = DVASPECT_CONTENT;
        fetc.lindex = -1;
        fetc.tymed = TYMED_HGLOBAL;
        const char *format_mime = "CF_HDROP";
        if (SUCCEEDED(pDataObject->lpVtbl->QueryGetData(pDataObject, &fetc))) {
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Drop File for QueryGetData, format %08x '%s', success",
                         fetc.cfFormat, format_mime);
            STGMEDIUM med;
            HRESULT hres = pDataObject->lpVtbl->GetData(pDataObject, &fetc, &med);
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Drop File for      GetData, format %08x '%s', HRESULT is %08lx",
                         fetc.cfFormat, format_mime, hres);
            if (SUCCEEDED(hres)) {
                const size_t bsize = GlobalSize(med.hGlobal);
                HDROP drop = (HDROP)GlobalLock(med.hGlobal);
                SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                             ". In Drop File for   GlobalLock, format %08x '%s', memory (%lu) %p",
                             fetc.cfFormat, format_mime, (unsigned long)bsize, drop);
                UINT count = DragQueryFile(drop, 0xFFFFFFFF, NULL, 0);
                for (UINT i = 0; i < count; ++i) {
                    UINT size = DragQueryFile(drop, i, NULL, 0) + 1;
                    LPTSTR buffer = (LPTSTR)SDL_malloc(size * sizeof(TCHAR));
                    if (buffer) {
                        if (DragQueryFile(drop, i, buffer, size)) {
                            char *file = WIN_StringToUTF8(buffer);
                            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                                         ". In Drop File, file (%lu of %lu) '%s'",
                                         (unsigned long)SDL_strlen(file), (unsigned long)bsize, file);
                            SDL_SendDropFile(target->window, NULL, file);
                            SDL_free(file);
                        }
                        SDL_free(buffer);
                    }
                }
                GlobalUnlock(med.hGlobal);
                ReleaseStgMedium(&med);
                SDL_SendDropComplete(target->window);
                return S_OK;
            }
        }
    }

    SDL_SendDropComplete(target->window);
    return S_OK;
}

static void *vtDropTarget[] = {
    (void *)(SDLDropTarget_QueryInterface),
    (void *)(SDLDropTarget_AddRef),
    (void *)(SDLDropTarget_Release),
    (void *)(SDLDropTarget_DragEnter),
    (void *)(SDLDropTarget_DragOver),
    (void *)(SDLDropTarget_DragLeave),
    (void *)(SDLDropTarget_Drop)
};

void WIN_AcceptDragAndDrop(SDL_Window *window, bool accept)
{
    SDL_WindowData *data = window->internal;
    if (data->videodata->oleinitialized) {
        if (accept && !data->drop_target) {
            SDLDropTarget *drop_target = (SDLDropTarget *)SDL_calloc(1, sizeof(SDLDropTarget));
            if (drop_target != NULL) {
                drop_target->lpVtbl = vtDropTarget;
                drop_target->window = window;
                drop_target->hwnd = data->hwnd;
                drop_target->format_file = RegisterClipboardFormat(L"text/uri-list");
                drop_target->format_text = RegisterClipboardFormat(L"text/plain;charset=utf-8");
                data->drop_target = drop_target;
                SDLDropTarget_AddRef(drop_target);
                RegisterDragDrop(data->hwnd, (LPDROPTARGET)drop_target);
                SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                             ". In Accept Drag and Drop, window %u, enabled Full OLE IDropTarget",
                             window->id);
            }
        } else if (!accept && data->drop_target) {
            RevokeDragDrop(data->hwnd);
            SDLDropTarget_Release(data->drop_target);
            data->drop_target = NULL;
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In Accept Drag and Drop, window %u, disabled Full OLE IDropTarget",
                         window->id);
        }
    } else {
        DragAcceptFiles(data->hwnd, accept ? TRUE : FALSE);
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In Accept Drag and Drop, window %u, %s Fallback WM_DROPFILES",
                     window->id, (accept ? "enabled" : "disabled"));
    }
}

bool WIN_FlashWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_FlashOperation operation)
{
    FLASHWINFO desc;

    SDL_zero(desc);
    desc.cbSize = sizeof(desc);
    desc.hwnd = window->internal->hwnd;
    switch (operation) {
    case SDL_FLASH_CANCEL:
        desc.dwFlags = FLASHW_STOP;
        break;
    case SDL_FLASH_BRIEFLY:
        desc.dwFlags = FLASHW_TRAY;
        desc.uCount = 1;
        break;
    case SDL_FLASH_UNTIL_FOCUSED:
        desc.dwFlags = (FLASHW_TRAY | FLASHW_TIMERNOFG);
        break;
    default:
        return SDL_Unsupported();
    }

    FlashWindowEx(&desc);

    return true;
}

bool WIN_ApplyWindowProgress(SDL_VideoDevice *_this, SDL_Window* window)
{
#ifdef HAVE_SHOBJIDL_CORE_H
    SDL_WindowData *data = window->internal;
    if (!data->taskbar_button_created) {
        return true;
    }

    if (window->progress_state == SDL_PROGRESS_STATE_NONE && !data->videodata->taskbar_list) {
        return true;
    }

    ITaskbarList3 *taskbar_list = GetTaskbarList(window);
    if (!taskbar_list) {
        return false;
    }

    TBPFLAG tbpFlags;
    switch (window->progress_state) {
    case SDL_PROGRESS_STATE_NONE:
        tbpFlags = TBPF_NOPROGRESS;
        break;
    case SDL_PROGRESS_STATE_INDETERMINATE:
        tbpFlags = TBPF_INDETERMINATE;
        break;
    case SDL_PROGRESS_STATE_NORMAL:
        tbpFlags = TBPF_NORMAL;
        break;
    case SDL_PROGRESS_STATE_PAUSED:
        tbpFlags = TBPF_PAUSED;
        break;
    case SDL_PROGRESS_STATE_ERROR:
        tbpFlags = TBPF_ERROR;
        break;
    default:
        return SDL_SetError("Parameter 'state' is not supported");
    }

    HRESULT ret = taskbar_list->lpVtbl->SetProgressState(taskbar_list, data->hwnd, tbpFlags);
    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("ITaskbarList3::SetProgressState()", ret);
    }

    if (window->progress_state >= SDL_PROGRESS_STATE_NORMAL) {
        ret = taskbar_list->lpVtbl->SetProgressValue(taskbar_list, data->hwnd, (ULONGLONG)(window->progress_value * 10000.f), 10000);
        if (FAILED(ret)) {
            return WIN_SetErrorFromHRESULT("ITaskbarList3::SetProgressValue()", ret);
        }
    }
#endif
    return true;
}

void WIN_ShowWindowSystemMenu(SDL_Window *window, int x, int y)
{
    const SDL_WindowData *data = window->internal;
    POINT pt;

    pt.x = x;
    pt.y = y;
    ClientToScreen(data->hwnd, &pt);
    SendMessage(data->hwnd, WM_POPUPSYSTEMMENU, 0, MAKELPARAM(pt.x, pt.y));
}

bool WIN_SetWindowFocusable(SDL_VideoDevice *_this, SDL_Window *window, bool focusable)
{
    if (!SDL_WINDOW_IS_POPUP(window)) {
        SDL_WindowData *data = window->internal;
        HWND hwnd = data->hwnd;
        const LONG style = GetWindowLong(hwnd, GWL_EXSTYLE);

        SDL_assert(style != 0);

        if (focusable) {
            if (style & WS_EX_NOACTIVATE) {
                if (SetWindowLong(hwnd, GWL_EXSTYLE, style & ~WS_EX_NOACTIVATE) == 0) {
                    return WIN_SetError("SetWindowLong()");
                }
            }
        } else {
            if (!(style & WS_EX_NOACTIVATE)) {
                if (SetWindowLong(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE) == 0) {
                    return WIN_SetError("SetWindowLong()");
                }
            }
        }
    } else if (window->flags & SDL_WINDOW_POPUP_MENU) {
        if (!(window->flags & SDL_WINDOW_HIDDEN)) {
            if (!focusable && (window->flags & SDL_WINDOW_INPUT_FOCUS)) {
                SDL_Window *new_focus;
                const bool set_focus = SDL_ShouldRelinquishPopupFocus(window, &new_focus);
                WIN_SetKeyboardFocus(new_focus, set_focus);
            } else if (focusable) {
                if (SDL_ShouldFocusPopup(window)) {
                    WIN_SetKeyboardFocus(window, true);
                }
            }
        }

        return true;
    }

    return true;
}
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

void WIN_UpdateDarkModeForHWND(HWND hwnd)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    SDL_SharedObject *ntdll = SDL_LoadObject("ntdll.dll");
    if (!ntdll) {
        return;
    }
    // There is no function to get Windows build number, so let's get it here via RtlGetVersion
    RtlGetVersion_t RtlGetVersionFunc = (RtlGetVersion_t)SDL_LoadFunction(ntdll, "RtlGetVersion");
    NT_OSVERSIONINFOW os_info;
    os_info.dwOSVersionInfoSize = sizeof(NT_OSVERSIONINFOW);
    os_info.dwBuildNumber = 0;
    if (RtlGetVersionFunc) {
        RtlGetVersionFunc(&os_info);
    }
    SDL_UnloadObject(ntdll);
    os_info.dwBuildNumber &= ~0xF0000000;
    if (os_info.dwBuildNumber < 17763) {
        // Too old to support dark mode
        return;
    }
    SDL_SharedObject *uxtheme = SDL_LoadObject("uxtheme.dll");
    if (!uxtheme) {
        return;
    }
    RefreshImmersiveColorPolicyState_t RefreshImmersiveColorPolicyStateFunc = (RefreshImmersiveColorPolicyState_t)SDL_LoadFunction(uxtheme, MAKEINTRESOURCEA(104));
    ShouldAppsUseDarkMode_t ShouldAppsUseDarkModeFunc = (ShouldAppsUseDarkMode_t)SDL_LoadFunction(uxtheme, MAKEINTRESOURCEA(132));
    AllowDarkModeForWindow_t AllowDarkModeForWindowFunc = (AllowDarkModeForWindow_t)SDL_LoadFunction(uxtheme, MAKEINTRESOURCEA(133));
    if (os_info.dwBuildNumber < 18362) {
        AllowDarkModeForApp_t AllowDarkModeForAppFunc = (AllowDarkModeForApp_t)SDL_LoadFunction(uxtheme, MAKEINTRESOURCEA(135));
        if (AllowDarkModeForAppFunc) {
            AllowDarkModeForAppFunc(true);
        }
    } else {
        SetPreferredAppMode_t SetPreferredAppModeFunc = (SetPreferredAppMode_t)SDL_LoadFunction(uxtheme, MAKEINTRESOURCEA(135));
        if (SetPreferredAppModeFunc) {
            SetPreferredAppModeFunc(UXTHEME_APPMODE_ALLOW_DARK);
        }
    }
    if (RefreshImmersiveColorPolicyStateFunc) {
        RefreshImmersiveColorPolicyStateFunc();
    }
    if (AllowDarkModeForWindowFunc) {
        AllowDarkModeForWindowFunc(hwnd, true);
    }
    BOOL value;
    // Check dark mode using ShouldAppsUseDarkMode, but use SDL_GetSystemTheme as a fallback
    if (ShouldAppsUseDarkModeFunc) {
        value = ShouldAppsUseDarkModeFunc() ? TRUE : FALSE;
    } else {
        value = (SDL_GetSystemTheme() == SDL_SYSTEM_THEME_DARK) ? TRUE : FALSE;
    }
    SDL_UnloadObject(uxtheme);
    if (os_info.dwBuildNumber < 18362) {
        SetProp(hwnd, TEXT("UseImmersiveDarkModeColors"), SDL_reinterpret_cast(HANDLE, SDL_static_cast(INT_PTR, value)));
    } else {
        HMODULE user32 = GetModuleHandle(TEXT("user32.dll"));
        if (user32) {
            SetWindowCompositionAttribute_t SetWindowCompositionAttributeFunc = (SetWindowCompositionAttribute_t)GetProcAddress(user32, "SetWindowCompositionAttribute");
            if (SetWindowCompositionAttributeFunc) {
                WINDOWCOMPOSITIONATTRIBDATA data = { WCA_USEDARKMODECOLORS, &value, sizeof(value) };
                SetWindowCompositionAttributeFunc(hwnd, &data);
            }
        }
    }
#endif
}

bool WIN_SetWindowParent(SDL_VideoDevice *_this, SDL_Window *window, SDL_Window *parent)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    SDL_WindowData *child_data = window->internal;
    const LONG_PTR parent_hwnd = (LONG_PTR)(parent ? parent->internal->hwnd : NULL);
    const DWORD style = GetWindowLong(child_data->hwnd, GWL_STYLE);

    if (!(style & WS_CHILD)) {
        /* Despite the name, this changes the *owner* of a toplevel window, not
         * the parent of a child window.
         *
         * https://devblogs.microsoft.com/oldnewthing/20100315-00/?p=14613
         */
        SetWindowLongPtr(child_data->hwnd, GWLP_HWNDPARENT, parent_hwnd);
    } else {
        SetParent(child_data->hwnd, (HWND)parent_hwnd);
    }
#endif /*!defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)*/

    return true;
}

bool WIN_SetWindowModal(SDL_VideoDevice *_this, SDL_Window *window, bool modal)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    const HWND parent_hwnd = window->parent->internal->hwnd;

    if (modal) {
        // Disable the parent window.
        EnableWindow(parent_hwnd, FALSE);
    } else if (!(window->flags & SDL_WINDOW_HIDDEN)) {
        // Re-enable the parent window
        EnableWindow(parent_hwnd, TRUE);
    }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

    return true;
}

#endif // SDL_VIDEO_DRIVER_WINDOWS
