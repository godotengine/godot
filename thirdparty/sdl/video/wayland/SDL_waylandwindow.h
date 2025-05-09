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

#ifndef SDL_waylandwindow_h_
#define SDL_waylandwindow_h_

#include "../SDL_sysvideo.h"
#include "../../events/SDL_touch_c.h"

#include "SDL_waylandvideo.h"
#include "SDL_waylandshmbuffer.h"

struct SDL_WindowData
{
    SDL_Window *sdlwindow;
    SDL_VideoData *waylandData;
    struct wl_surface *surface;
    struct wl_callback *gles_swap_frame_callback;
    struct wl_event_queue *gles_swap_frame_event_queue;
    struct wl_surface *gles_swap_frame_surface_wrapper;
    struct wl_callback *surface_frame_callback;

    union
    {
#ifdef HAVE_LIBDECOR_H
        struct
        {
            struct libdecor_frame *frame;
            bool initial_configure_seen;
        } libdecor;
#endif
        struct
        {
            struct xdg_surface *surface;
            union
            {
                struct
                {
                    struct xdg_toplevel *xdg_toplevel;
                } toplevel;
                struct
                {
                    struct xdg_popup *xdg_popup;
                    struct xdg_positioner *xdg_positioner;
                } popup;
            };
            bool initial_configure_seen;
        } xdg;
    } shell_surface;
    enum
    {
        WAYLAND_SHELL_SURFACE_TYPE_UNKNOWN = 0,
        WAYLAND_SHELL_SURFACE_TYPE_XDG_TOPLEVEL,
        WAYLAND_SHELL_SURFACE_TYPE_XDG_POPUP,
        WAYLAND_SHELL_SURFACE_TYPE_LIBDECOR,
        WAYLAND_SHELL_SURFACE_TYPE_CUSTOM
    } shell_surface_type;
    enum
    {
        WAYLAND_SHELL_SURFACE_STATUS_HIDDEN = 0,
        WAYLAND_SHELL_SURFACE_STATUS_WAITING_FOR_CONFIGURE,
        WAYLAND_SHELL_SURFACE_STATUS_WAITING_FOR_FRAME,
        WAYLAND_SHELL_SURFACE_STATUS_SHOW_PENDING,
        WAYLAND_SHELL_SURFACE_STATUS_SHOWN
    } shell_surface_status;
    enum
    {
        WAYLAND_WM_CAPS_WINDOW_MENU = 0x01,
        WAYLAND_WM_CAPS_MAXIMIZE = 0x02,
        WAYLAND_WM_CAPS_FULLSCREEN = 0x04,
        WAYLAND_WM_CAPS_MINIMIZE = 0x08,

        WAYLAND_WM_CAPS_ALL = WAYLAND_WM_CAPS_WINDOW_MENU |
                              WAYLAND_WM_CAPS_MAXIMIZE |
                              WAYLAND_WM_CAPS_FULLSCREEN |
                              WAYLAND_WM_CAPS_MINIMIZE
    } wm_caps;
    enum
    {
        WAYLAND_TOPLEVEL_CONSTRAINED_LEFT = 0x01,
        WAYLAND_TOPLEVEL_CONSTRAINED_RIGHT = 0x02,
        WAYLAND_TOPLEVEL_CONSTRAINED_TOP = 0x04,
        WAYLAND_TOPLEVEL_CONSTRAINED_BOTTOM = 0x08
    } toplevel_constraints;

    struct wl_egl_window *egl_window;
#ifdef SDL_VIDEO_OPENGL_EGL
    EGLSurface egl_surface;
#endif
    struct zxdg_toplevel_decoration_v1 *server_decoration;
    struct zwp_idle_inhibitor_v1 *idle_inhibitor;
    struct xdg_activation_token_v1 *activation_token;
    struct wp_viewport *viewport;
    struct wp_fractional_scale_v1 *fractional_scale;
    struct zxdg_exported_v2 *exported;
    struct xdg_dialog_v1 *xdg_dialog_v1;
    struct wp_alpha_modifier_surface_v1 *wp_alpha_modifier_surface_v1;
    struct xdg_toplevel_icon_v1 *xdg_toplevel_icon_v1;
    struct frog_color_managed_surface *frog_color_managed_surface;
    struct wp_color_management_surface_feedback_v1 *wp_color_management_surface_feedback;

    struct Wayland_ColorInfoState *color_info_state;

    SDL_AtomicInt swap_interval_ready;

    SDL_DisplayData **outputs;
    int num_outputs;

    char *app_id;
    double scale_factor;

    struct Wayland_SHMBuffer *icon_buffers;
    int icon_buffer_count;

    // Keyboard and pointer focus refcount.
    int keyboard_focus_count;
    int pointer_focus_count;

    struct
    {
        double x;
        double y;
    } pointer_scale;

    // The in-flight window size request.
    struct
    {
        // The requested logical window size.
        int logical_width;
        int logical_height;

        // The size of the window in pixels, when using screen space scaling.
        int pixel_width;
        int pixel_height;
    } requested;

    // The current size of the window and drawable backing store.
    struct
    {
        // The size of the underlying window.
        int logical_width;
        int logical_height;

        // The size of the window backbuffer in pixels.
        int pixel_width;
        int pixel_height;
    } current;

    // The last compositor requested parameters; used for deduplication of window geometry configuration.
    struct
    {
        int width;
        int height;
    } last_configure;

    // System enforced window size limits.
    struct
    {
        // Minimum allowed logical window size.
        int min_width;
        int min_height;
    } system_limits;

    struct
    {
        int width;
        int height;
    } toplevel_bounds;

    struct
    {
        int hint;
        int purpose;
        bool active;
    } text_input_props;

    SDL_DisplayID last_displayID;
    int fullscreen_deadline_count;
    int maximized_restored_deadline_count;
    Uint64 last_focus_event_time_ns;
    int icc_fd;
    Uint32 icc_size;
    bool floating;
    bool suspended;
    bool resizing;
    bool active;
    bool drop_interactive_resizes;
    bool is_fullscreen;
    bool fullscreen_exclusive;
    bool drop_fullscreen_requests;
    bool showing_window;
    bool fullscreen_was_positioned;
    bool show_hide_sync_required;
    bool scale_to_display;
    bool reparenting_required;
    bool double_buffer;

    SDL_HitTestResult hit_test_result;

    struct wl_list external_window_list_link;
};

extern void Wayland_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void Wayland_HideWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void Wayland_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern SDL_FullscreenResult Wayland_SetWindowFullscreen(SDL_VideoDevice *_this, SDL_Window *window, SDL_VideoDisplay *_display, SDL_FullscreenOp fullscreen);
extern void Wayland_MaximizeWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void Wayland_MinimizeWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern bool Wayland_SetWindowMouseRect(SDL_VideoDevice *_this, SDL_Window *window);
extern bool Wayland_SetWindowMouseGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed);
extern bool Wayland_SetWindowKeyboardGrab(SDL_VideoDevice *_this, SDL_Window *window, bool grabbed);
extern void Wayland_RestoreWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern void Wayland_SetWindowBordered(SDL_VideoDevice *_this, SDL_Window *window, bool bordered);
extern void Wayland_SetWindowResizable(SDL_VideoDevice *_this, SDL_Window *window, bool resizable);
extern bool Wayland_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props);
extern bool Wayland_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window);
extern void Wayland_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window);
extern void Wayland_SetWindowMinimumSize(SDL_VideoDevice *_this, SDL_Window *window);
extern void Wayland_SetWindowMaximumSize(SDL_VideoDevice *_this, SDL_Window *window);
extern void Wayland_GetWindowSizeInPixels(SDL_VideoDevice *_this, SDL_Window *window, int *w, int *h);
extern SDL_DisplayID Wayland_GetDisplayForWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern bool Wayland_SetWindowParent(SDL_VideoDevice *_this, SDL_Window *window, SDL_Window *parent_window);
extern bool Wayland_SetWindowModal(SDL_VideoDevice *_this, SDL_Window *window, bool modal);
extern bool Wayland_SetWindowOpacity(SDL_VideoDevice *_this, SDL_Window *window, float opacity);
extern void Wayland_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window);
extern void Wayland_ShowWindowSystemMenu(SDL_Window *window, int x, int y);
extern void Wayland_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window);
extern bool Wayland_SuspendScreenSaver(SDL_VideoDevice *_this);
extern bool Wayland_SetWindowIcon(SDL_VideoDevice *_this, SDL_Window *window, SDL_Surface *icon);
extern bool Wayland_SetWindowFocusable(SDL_VideoDevice *_this, SDL_Window *window, bool focusable);
extern float Wayland_GetWindowContentScale(SDL_VideoDevice *_this, SDL_Window *window);
extern void *Wayland_GetWindowICCProfile(SDL_VideoDevice *_this, SDL_Window *window, size_t *size);

extern bool Wayland_SetWindowHitTest(SDL_Window *window, bool enabled);
extern bool Wayland_FlashWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_FlashOperation operation);
extern bool Wayland_SyncWindow(SDL_VideoDevice *_this, SDL_Window *window);

extern void Wayland_RemoveOutputFromWindow(SDL_WindowData *window, SDL_DisplayData *display_data);

#endif // SDL_waylandwindow_h_
