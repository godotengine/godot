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

#ifndef SDL_windowswindow_h_
#define SDL_windowswindow_h_

#ifdef SDL_VIDEO_OPENGL_EGL
#include "../SDL_egl_c.h"
#else
#include "../SDL_sysvideo.h"
#endif

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

typedef enum SDL_WindowRect
{
    SDL_WINDOWRECT_CURRENT,
    SDL_WINDOWRECT_WINDOWED,
    SDL_WINDOWRECT_FLOATING,
    SDL_WINDOWRECT_PENDING
} SDL_WindowRect;

typedef enum SDL_WindowEraseBackgroundMode
{
    SDL_ERASEBACKGROUNDMODE_NEVER,
    SDL_ERASEBACKGROUNDMODE_INITIAL,
    SDL_ERASEBACKGROUNDMODE_ALWAYS,
} SDL_WindowEraseBackgroundMode;

typedef struct
{
    void **lpVtbl;
    int refcount;
    SDL_Window *window;
    HWND hwnd;
    UINT format_text;
    UINT format_file;
} SDLDropTarget;

struct SDL_WindowData
{
    SDL_Window *window;
    HWND hwnd;
    HWND parent;
    HDC hdc;
    HDC mdc;
    HINSTANCE hinstance;
    HBITMAP hbm;
    WNDPROC wndproc;
    HHOOK keyboard_hook;
    WPARAM mouse_button_flags;
    LPARAM last_pointer_update;
    WCHAR high_surrogate;
    bool initializing;
    bool expected_resize;
    bool in_border_change;
    bool in_title_click;
    Uint8 focus_click_pending;
    bool postpone_clipcursor;
    bool clipcursor_queued;
    bool windowed_mode_was_maximized;
    bool in_window_deactivation;
    bool force_ws_maximizebox;
    bool disable_move_size_events;
    int in_modal_loop;
    RECT initial_size_rect;
    RECT cursor_clipped_rect; // last successfully committed clipping rect for this window
    RECT cursor_ctrlock_rect; // this is Windows-specific, but probably does not need to be per-window
    bool mouse_tracked;
    bool destroy_parent_with_window;
    SDL_DisplayID last_displayID;
    WCHAR *ICMFileName;
    SDL_WindowEraseBackgroundMode hint_erase_background_mode;
    bool taskbar_button_created;
    struct SDL_VideoData *videodata;
#ifdef SDL_VIDEO_OPENGL_EGL
    EGLSurface egl_surface;
#endif

    // Whether we retain the content of the window when changing state
    UINT copybits_flag;
    SDLDropTarget *drop_target;
};

extern bool WIN_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props);
extern void WIN_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window);
extern bool WIN_SetWindowIcon(SDL_VideoDevice *_this, SDL_Window *window, SDL_Surface *icon);
extern bool WIN_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window);
extern void WIN_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window);
extern bool WIN_GetWindowBordersSize(SDL_VideoDevice *_this, SDL_Window *window, int *top, int *left, int *bottom, int *right);
extern void WIN_GetWindowSizeInPixels(SDL_VideoDevice *_this, SDL_Window *window, int *width, int *height);
extern bool WIN_SetWindowOpacity(SDL_VideoDevice *_this, SDL_Window *window, float opacity);
extern void WIN_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void WIN_HideWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void WIN_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void WIN_MaximizeWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void WIN_MinimizeWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void WIN_RestoreWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void WIN_SetWindowBordered(SDL_VideoDevice *_this, SDL_Window *window, bool bordered);
extern void WIN_SetWindowResizable(SDL_VideoDevice *_this, SDL_Window *window, bool resizable);
extern void WIN_SetWindowAlwaysOnTop(SDL_VideoDevice *_this, SDL_Window *window, bool on_top);
extern SDL_FullscreenResult WIN_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *display, SDL_FullscreenOp fullscreen);
extern void WIN_UpdateWindowICCProfile(SDL_Window *window, bool send_event);
extern void *WIN_GetWindowICCProfile(SDL_VideoDevice *_this, SDL_Window *window, size_t *size);
extern bool WIN_SetWindowMouseRect(SDL_VideoDevice *_this, SDL_Window *window);
extern bool WIN_SetWindowMouseGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed);
extern bool WIN_SetWindowKeyboardGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed);
extern void WIN_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void WIN_OnWindowEnter(SDL_VideoDevice *_this, SDL_Window *window);
extern void WIN_UpdateClipCursor(SDL_Window *window);
extern void WIN_UnclipCursorForWindow(SDL_Window *window);
extern bool WIN_SetWindowHitTest(SDL_Window *window, bool enabled);
extern void WIN_AcceptDragAndDrop(SDL_Window *window, bool accept);
extern bool WIN_FlashWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_FlashOperation operation);
extern bool WIN_ApplyWindowProgress(SDL_VideoDevice *_this, SDL_Window *window);
extern void WIN_UpdateDarkModeForHWND(HWND hwnd);
extern bool WIN_SetWindowPositionInternal(SDL_Window *window, UINT flags, SDL_WindowRect rect_type);
extern void WIN_ShowWindowSystemMenu(SDL_Window *window, int x, int y);
extern bool WIN_SetWindowFocusable(SDL_VideoDevice *_this, SDL_Window *window, bool focusable);
extern bool WIN_AdjustWindowRect(SDL_Window *window, int *x, int *y, int *width, int *height, SDL_WindowRect rect_type);
extern bool WIN_AdjustWindowRectForHWND(HWND hwnd, LPRECT lpRect, UINT frame_dpi);
extern bool WIN_SetWindowParent(SDL_VideoDevice *_this, SDL_Window *window, SDL_Window *parent);
extern bool WIN_SetWindowModal(SDL_VideoDevice *_this, SDL_Window *window, bool modal);

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#endif // SDL_windowswindow_h_
