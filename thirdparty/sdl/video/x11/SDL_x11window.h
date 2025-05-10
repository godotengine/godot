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

#ifndef SDL_x11window_h_
#define SDL_x11window_h_

/* We need to queue the focus in/out changes because they may occur during
   video mode changes and we can respond to them by triggering more mode
   changes.
*/
#define PENDING_FOCUS_TIME 200

#ifdef SDL_VIDEO_OPENGL_EGL
#include <EGL/egl.h>
#endif

typedef enum
{
    PENDING_FOCUS_NONE,
    PENDING_FOCUS_IN,
    PENDING_FOCUS_OUT
} PendingFocusEnum;

struct SDL_WindowData
{
    SDL_Window *window;
    Window xwindow;
    Visual *visual;
    Colormap colormap;
#ifndef NO_SHARED_MEMORY
    // MIT shared memory extension information
    bool use_mitshm;
    XShmSegmentInfo shminfo;
#endif
    XImage *ximage;
    GC gc;
    XIC ic;
    bool created;
    int border_left;
    int border_right;
    int border_top;
    int border_bottom;
    bool xinput2_mouse_enabled;
    bool xinput2_keyboard_enabled;
    bool mouse_grabbed;
    Uint64 last_focus_event_time;
    PendingFocusEnum pending_focus;
    Uint64 pending_focus_time;
    bool pending_move;
    SDL_Point pending_move_point;
    XConfigureEvent last_xconfigure;
    XConfigureEvent pending_xconfigure;
    struct SDL_VideoData *videodata;
    unsigned long user_time;
    Atom xdnd_req;
    Window xdnd_source;
    bool flashing_window;
    Uint64 flash_cancel_time;
#ifdef SDL_VIDEO_OPENGL_EGL
    EGLSurface egl_surface;
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
    bool pointer_barrier_active;
    PointerBarrier barrier[4];
    SDL_Rect barrier_rect;
#endif // SDL_VIDEO_DRIVER_X11_XFIXES
#ifdef SDL_VIDEO_DRIVER_X11_XSYNC
    XSyncCounter resize_counter;
    XSyncValue resize_id;
    bool resize_in_progress;
#endif /* SDL_VIDEO_DRIVER_X11_XSYNC */

    SDL_Rect expected;
    SDL_DisplayMode requested_fullscreen_mode;

    enum
    {
        X11_PENDING_OP_NONE = 0x00,
        X11_PENDING_OP_RESTORE = 0x01,
        X11_PENDING_OP_MINIMIZE = 0x02,
        X11_PENDING_OP_MAXIMIZE = 0x04,
        X11_PENDING_OP_FULLSCREEN = 0x08,
        X11_PENDING_OP_MOVE = 0x10,
        X11_PENDING_OP_RESIZE = 0x20
    } pending_operation;

    enum
    {
        X11_SIZE_MOVE_EVENTS_DISABLE = 0x01, // Events are completely disabled.
        X11_SIZE_MOVE_EVENTS_WAIT_FOR_BORDERS = 0x02, // Events are disabled until a _NET_FRAME_EXTENTS event arrives.
    } size_move_event_flags;

    bool pending_size;
    bool pending_position;
    bool window_was_maximized;
    bool previous_borders_nonzero;
    bool toggle_borders;
    bool fullscreen_borders_forced_on;
    bool was_shown;
    bool emit_size_move_after_property_notify;
    bool tracking_mouse_outside_window;
    SDL_HitTestResult hit_test_result;

    XPoint xim_spot;
    char *preedit_text;
    XIMFeedback *preedit_feedback;
    int preedit_length;
    int preedit_cursor;
    bool ime_needs_clear_composition;
};

extern void X11_SetNetWMState(SDL_VideoDevice *_this, Window xwindow, SDL_WindowFlags flags);
extern Uint32 X11_GetNetWMState(SDL_VideoDevice *_this, SDL_Window *window, Window xwindow);

extern bool X11_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props);
extern char *X11_GetWindowTitle(SDL_VideoDevice *_this, Window xwindow);
extern void X11_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window);
extern bool X11_SetWindowIcon(SDL_VideoDevice *_this, SDL_Window *window, SDL_Surface *icon);
extern bool X11_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window);
extern void X11_SetWindowMinimumSize(SDL_VideoDevice *_this, SDL_Window *window);
extern void X11_SetWindowMaximumSize(SDL_VideoDevice *_this, SDL_Window *window);
extern void X11_SetWindowAspectRatio(SDL_VideoDevice *_this, SDL_Window *window);
extern bool X11_GetWindowBordersSize(SDL_VideoDevice *_this, SDL_Window *window, int *top, int *left, int *bottom, int *right);
extern bool X11_SetWindowOpacity(SDL_VideoDevice *_this, SDL_Window *window, float opacity);
extern bool X11_SetWindowParent(SDL_VideoDevice *_this, SDL_Window *window, SDL_Window *parent);
extern bool X11_SetWindowModal(SDL_VideoDevice *_this, SDL_Window *window, bool modal);
extern void X11_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window);
extern void X11_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void X11_HideWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void X11_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void X11_MaximizeWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void X11_MinimizeWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void X11_RestoreWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void X11_SetWindowBordered(SDL_VideoDevice *_this, SDL_Window *window, bool bordered);
extern void X11_SetWindowResizable(SDL_VideoDevice *_this, SDL_Window *window, bool resizable);
extern void X11_SetWindowAlwaysOnTop(SDL_VideoDevice *_this, SDL_Window *window, bool on_top);
extern SDL_FullscreenResult X11_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *display, SDL_FullscreenOp fullscreen);
extern void *X11_GetWindowICCProfile(SDL_VideoDevice *_this, SDL_Window *window, size_t *size);
extern bool X11_SetWindowMouseGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed);
extern bool X11_SetWindowKeyboardGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed);
extern void X11_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern bool X11_SetWindowHitTest(SDL_Window *window, bool enabled);
extern void X11_AcceptDragAndDrop(SDL_Window *window, bool accept);
extern bool X11_FlashWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_FlashOperation operation);
extern void X11_ShowWindowSystemMenu(SDL_Window *window, int x, int y);
extern bool X11_SyncWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern bool X11_SetWindowFocusable(SDL_VideoDevice *_this, SDL_Window *window, bool focusable);

extern bool SDL_X11_SetWindowTitle(Display *display, Window xwindow, char *title);
extern void X11_UpdateWindowPosition(SDL_Window *window, bool use_current_position);
extern void X11_SetWindowMinMax(SDL_Window *window, bool use_current);

#endif // SDL_x11window_h_
