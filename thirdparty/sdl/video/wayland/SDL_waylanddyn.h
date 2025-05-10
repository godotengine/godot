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
#ifndef SDL_waylanddyn_h_
#define SDL_waylanddyn_h_

#include "SDL_internal.h"

/* We can't include wayland-client.h here
 * but we need some structs from it
 */
struct wl_interface;
struct wl_proxy;
struct wl_event_queue;
struct wl_display;
struct wl_surface;
struct wl_shm;

// We also need some for libdecor
struct wl_seat;
struct wl_output;
struct libdecor;
struct libdecor_frame;
struct libdecor_state;
struct libdecor_configuration;
struct libdecor_interface;
struct libdecor_frame_interface;
enum libdecor_resize_edge;
enum libdecor_capabilities;
enum libdecor_window_state;

#include "wayland-cursor.h"
#include "wayland-util.h"
#include "xkbcommon/xkbcommon.h"
#include "xkbcommon/xkbcommon-compose.h"

// Must be included before our #defines, see Bugzilla #4957
#include "wayland-client-core.h"

#define SDL_WAYLAND_CHECK_VERSION(x, y, z)                        \
    (WAYLAND_VERSION_MAJOR > x ||                                 \
     (WAYLAND_VERSION_MAJOR == x && WAYLAND_VERSION_MINOR > y) || \
     (WAYLAND_VERSION_MAJOR == x && WAYLAND_VERSION_MINOR == y && WAYLAND_VERSION_MICRO >= z))

#ifdef HAVE_LIBDECOR_H
#define SDL_LIBDECOR_CHECK_VERSION(x, y, z)                                 \
    (SDL_LIBDECOR_VERSION_MAJOR > x ||                                      \
     (SDL_LIBDECOR_VERSION_MAJOR == x && SDL_LIBDECOR_VERSION_MINOR > y) || \
     (SDL_LIBDECOR_VERSION_MAJOR == x && SDL_LIBDECOR_VERSION_MINOR == y && SDL_LIBDECOR_VERSION_PATCH >= z))
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern bool SDL_WAYLAND_LoadSymbols(void);
extern void SDL_WAYLAND_UnloadSymbols(void);

#define SDL_WAYLAND_MODULE(modname) extern int SDL_WAYLAND_HAVE_##modname;
#define SDL_WAYLAND_SYM(rc, fn, params)        \
    typedef rc(*SDL_DYNWAYLANDFN_##fn) params; \
    extern SDL_DYNWAYLANDFN_##fn WAYLAND_##fn;
#define SDL_WAYLAND_SYM_OPT(rc, fn, params)    \
    typedef rc(*SDL_DYNWAYLANDFN_##fn) params; \
    extern SDL_DYNWAYLANDFN_##fn WAYLAND_##fn;
#define SDL_WAYLAND_INTERFACE(iface) extern const struct wl_interface *WAYLAND_##iface;
#include "SDL_waylandsym.h"

#ifdef __cplusplus
}
#endif

#ifdef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC

#if defined(_WAYLAND_CLIENT_H) || defined(WAYLAND_CLIENT_H)
#error Do not include wayland-client ahead of SDL_waylanddyn.h in dynamic loading mode
#endif

/* wayland-client-protocol.h included from wayland-client.h
 * has inline functions that require these to be defined in dynamic loading mode
 */

#define wl_proxy_create                        (*WAYLAND_wl_proxy_create)
#define wl_proxy_destroy                       (*WAYLAND_wl_proxy_destroy)
#define wl_proxy_marshal                       (*WAYLAND_wl_proxy_marshal)
#define wl_proxy_set_user_data                 (*WAYLAND_wl_proxy_set_user_data)
#define wl_proxy_get_user_data                 (*WAYLAND_wl_proxy_get_user_data)
#define wl_proxy_get_version                   (*WAYLAND_wl_proxy_get_version)
#define wl_proxy_add_listener                  (*WAYLAND_wl_proxy_add_listener)
#define wl_proxy_marshal_constructor           (*WAYLAND_wl_proxy_marshal_constructor)
#define wl_proxy_marshal_constructor_versioned (*WAYLAND_wl_proxy_marshal_constructor_versioned)
#define wl_proxy_set_tag                       (*WAYLAND_wl_proxy_set_tag)
#define wl_proxy_get_tag                       (*WAYLAND_wl_proxy_get_tag)
#define wl_proxy_marshal_flags                 (*WAYLAND_wl_proxy_marshal_flags)
#define wl_proxy_marshal_array_flags           (*WAYLAND_wl_proxy_marshal_array_flags)
#define wl_display_reconnect                   (*WAYLAND_wl_display_reconnect)

#define wl_seat_interface                (*WAYLAND_wl_seat_interface)
#define wl_surface_interface             (*WAYLAND_wl_surface_interface)
#define wl_shm_pool_interface            (*WAYLAND_wl_shm_pool_interface)
#define wl_buffer_interface              (*WAYLAND_wl_buffer_interface)
#define wl_registry_interface            (*WAYLAND_wl_registry_interface)
#define wl_region_interface              (*WAYLAND_wl_region_interface)
#define wl_pointer_interface             (*WAYLAND_wl_pointer_interface)
#define wl_keyboard_interface            (*WAYLAND_wl_keyboard_interface)
#define wl_compositor_interface          (*WAYLAND_wl_compositor_interface)
#define wl_output_interface              (*WAYLAND_wl_output_interface)
#define wl_shm_interface                 (*WAYLAND_wl_shm_interface)
#define wl_data_device_interface         (*WAYLAND_wl_data_device_interface)
#define wl_data_offer_interface          (*WAYLAND_wl_data_offer_interface)
#define wl_data_source_interface         (*WAYLAND_wl_data_source_interface)
#define wl_data_device_manager_interface (*WAYLAND_wl_data_device_manager_interface)

/*
 * These must be included before libdecor.h, otherwise the libdecor header
 * pulls in the system Wayland protocol headers instead of ours.
 */
#include "wayland-client-protocol.h"
#include "wayland-egl.h"

#ifdef HAVE_LIBDECOR_H
// Must be included before our defines
#include <libdecor.h>

#define libdecor_unref                          (*WAYLAND_libdecor_unref)
#define libdecor_new                            (*WAYLAND_libdecor_new)
#define libdecor_decorate                       (*WAYLAND_libdecor_decorate)
#define libdecor_frame_unref                    (*WAYLAND_libdecor_frame_unref)
#define libdecor_frame_set_title                (*WAYLAND_libdecor_frame_set_title)
#define libdecor_frame_set_app_id               (*WAYLAND_libdecor_frame_set_app_id)
#define libdecor_frame_set_max_content_size     (*WAYLAND_libdecor_frame_set_max_content_size)
#define libdecor_frame_get_max_content_size     (*WAYLAND_libdecor_frame_get_max_content_size)
#define libdecor_frame_set_min_content_size     (*WAYLAND_libdecor_frame_set_min_content_size)
#define libdecor_frame_get_min_content_size     (*WAYLAND_libdecor_frame_get_min_content_size)
#define libdecor_frame_resize                   (*WAYLAND_libdecor_frame_resize)
#define libdecor_frame_move                     (*WAYLAND_libdecor_frame_move)
#define libdecor_frame_commit                   (*WAYLAND_libdecor_frame_commit)
#define libdecor_frame_set_minimized            (*WAYLAND_libdecor_frame_set_minimized)
#define libdecor_frame_set_maximized            (*WAYLAND_libdecor_frame_set_maximized)
#define libdecor_frame_unset_maximized          (*WAYLAND_libdecor_frame_unset_maximized)
#define libdecor_frame_set_fullscreen           (*WAYLAND_libdecor_frame_set_fullscreen)
#define libdecor_frame_unset_fullscreen         (*WAYLAND_libdecor_frame_unset_fullscreen)
#define libdecor_frame_set_capabilities         (*WAYLAND_libdecor_frame_set_capabilities)
#define libdecor_frame_unset_capabilities       (*WAYLAND_libdecor_frame_unset_capabilities)
#define libdecor_frame_has_capability           (*WAYLAND_libdecor_frame_has_capability)
#define libdecor_frame_set_visibility           (*WAYLAND_libdecor_frame_set_visibility)
#define libdecor_frame_is_visible               (*WAYLAND_libdecor_frame_is_visible)
#define libdecor_frame_is_floating              (*WAYLAND_libdecor_frame_is_floating)
#define libdecor_frame_set_parent               (*WAYLAND_libdecor_frame_set_parent)
#define libdecor_frame_show_window_menu         (*WAYLAND_libdecor_frame_show_window_menu)
#define libdecor_frame_get_wm_capabilities      (*WAYLAND_libdecor_frame_get_wm_capabilities)
#define libdecor_frame_get_xdg_surface          (*WAYLAND_libdecor_frame_get_xdg_surface)
#define libdecor_frame_get_xdg_toplevel         (*WAYLAND_libdecor_frame_get_xdg_toplevel)
#define libdecor_frame_translate_coordinate     (*WAYLAND_libdecor_frame_translate_coordinate)
#define libdecor_frame_map                      (*WAYLAND_libdecor_frame_map)
#define libdecor_state_new                      (*WAYLAND_libdecor_state_new)
#define libdecor_state_free                     (*WAYLAND_libdecor_state_free)
#define libdecor_configuration_get_content_size (*WAYLAND_libdecor_configuration_get_content_size)
#define libdecor_configuration_get_window_state (*WAYLAND_libdecor_configuration_get_window_state)
#define libdecor_dispatch                       (*WAYLAND_libdecor_dispatch)
#endif

#else // SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC

/*
 * These must be included before libdecor.h, otherwise the libdecor header
 * pulls in the system Wayland protocol headers instead of ours.
 */
#include "wayland-client-protocol.h"
#include "wayland-egl.h"

#ifdef HAVE_LIBDECOR_H
#include <libdecor.h>
#endif

#endif // SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC

#endif // SDL_waylanddyn_h_
