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

#ifndef SDL_waylandvideo_h_
#define SDL_waylandvideo_h_

#include <EGL/egl.h>
#include "wayland-util.h"

#include "../SDL_sysvideo.h"
#include "../../core/linux/SDL_dbus.h"
#include "../../core/linux/SDL_ime.h"

struct xkb_context;
struct SDL_WaylandSeat;

typedef struct
{
    struct wl_cursor_theme *theme;
    int size;
} SDL_WaylandCursorTheme;

typedef struct
{
    struct wl_list link;
    char wl_output_name[];
} SDL_WaylandConnectorName;

struct SDL_VideoData
{
    bool initializing;
    struct wl_display *display;
    int display_disconnected;
    struct wl_registry *registry;
    struct wl_compositor *compositor;
    struct wl_shm *shm;
    SDL_WaylandCursorTheme *cursor_themes;
    int num_cursor_themes;
    struct
    {
        struct xdg_wm_base *xdg;
#ifdef HAVE_LIBDECOR_H
        struct libdecor *libdecor;
#endif
    } shell;
    struct zwp_relative_pointer_manager_v1 *relative_pointer_manager;
    struct zwp_pointer_constraints_v1 *pointer_constraints;
    struct wp_cursor_shape_manager_v1 *cursor_shape_manager;
    struct wl_data_device_manager *data_device_manager;
    struct zwp_primary_selection_device_manager_v1 *primary_selection_device_manager;
    struct zxdg_decoration_manager_v1 *decoration_manager;
    struct zwp_keyboard_shortcuts_inhibit_manager_v1 *key_inhibitor_manager;
    struct zwp_idle_inhibit_manager_v1 *idle_inhibit_manager;
    struct xdg_activation_v1 *activation_manager;
    struct zwp_text_input_manager_v3 *text_input_manager;
    struct zxdg_output_manager_v1 *xdg_output_manager;
    struct wp_viewporter *viewporter;
    struct wp_fractional_scale_manager_v1 *fractional_scale_manager;
    struct zwp_input_timestamps_manager_v1 *input_timestamps_manager;
    struct zxdg_exporter_v2 *zxdg_exporter_v2;
    struct xdg_wm_dialog_v1 *xdg_wm_dialog_v1;
    struct wp_alpha_modifier_v1 *wp_alpha_modifier_v1;
    struct xdg_toplevel_icon_manager_v1 *xdg_toplevel_icon_manager_v1;
    struct frog_color_management_factory_v1 *frog_color_management_factory_v1;
    struct wp_color_manager_v1 *wp_color_manager_v1;
    struct zwp_tablet_manager_v2 *tablet_manager;

    struct xkb_context *xkb_context;

    struct wl_list seat_list;
    struct SDL_WaylandSeat *last_implicit_grab_seat;
    struct SDL_WaylandSeat *last_incoming_data_offer_seat;
    struct SDL_WaylandSeat *last_incoming_primary_selection_seat;

    SDL_DisplayData **output_list;
    int output_count;
    int output_max;

    bool relative_mode_enabled;
    bool display_externally_owned;

    bool scale_to_display_enabled;
};

struct SDL_DisplayData
{
    SDL_VideoData *videodata;
    struct wl_output *output;
    struct zxdg_output_v1 *xdg_output;
    struct wp_color_management_output_v1 *wp_color_management_output;
    char *wl_output_name;
    double scale_factor;
    uint32_t registry_id;
    int logical_width, logical_height;
    int pixel_width, pixel_height;
    int x, y, refresh, transform;
    SDL_DisplayOrientation orientation;
    int physical_width_mm, physical_height_mm;
    bool has_logical_position, has_logical_size;
    bool running_colorspace_event_queue;
    SDL_HDROutputProperties HDR;
    SDL_DisplayID display;
    SDL_VideoDisplay placeholder;
    int wl_output_done_count;
    struct Wayland_ColorInfoState *color_info_state;
};

// Needed here to get wl_surface declaration, fixes GitHub#4594
#include "SDL_waylanddyn.h"

extern void SDL_WAYLAND_register_surface(struct wl_surface *surface);
extern void SDL_WAYLAND_register_output(struct wl_output *output);
extern bool SDL_WAYLAND_own_surface(struct wl_surface *surface);
extern bool SDL_WAYLAND_own_output(struct wl_output *output);

extern SDL_WindowData *Wayland_GetWindowDataForOwnedSurface(struct wl_surface *surface);
void Wayland_AddWindowDataToExternalList(SDL_WindowData *data);
void Wayland_RemoveWindowDataFromExternalList(SDL_WindowData *data);

extern bool Wayland_LoadLibdecor(SDL_VideoData *data, bool ignore_xdg);

extern bool Wayland_VideoReconnect(SDL_VideoDevice *_this);

#endif // SDL_waylandvideo_h_
