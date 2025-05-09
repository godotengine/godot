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

#ifdef SDL_VIDEO_DRIVER_WAYLAND

#include "../../core/linux/SDL_system_theme.h"
#include "../../events/SDL_events_c.h"

#include "SDL_waylandclipboard.h"
#include "SDL_waylandcolor.h"
#include "SDL_waylandevents_c.h"
#include "SDL_waylandkeyboard.h"
#include "SDL_waylandmessagebox.h"
#include "SDL_waylandmouse.h"
#include "SDL_waylandopengles.h"
#include "SDL_waylandvideo.h"
#include "SDL_waylandvulkan.h"
#include "SDL_waylandwindow.h"

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <xkbcommon/xkbcommon.h>

#include <wayland-util.h>

#include "alpha-modifier-v1-client-protocol.h"
#include "cursor-shape-v1-client-protocol.h"
#include "fractional-scale-v1-client-protocol.h"
#include "frog-color-management-v1-client-protocol.h"
#include "idle-inhibit-unstable-v1-client-protocol.h"
#include "input-timestamps-unstable-v1-client-protocol.h"
#include "keyboard-shortcuts-inhibit-unstable-v1-client-protocol.h"
#include "pointer-constraints-unstable-v1-client-protocol.h"
#include "primary-selection-unstable-v1-client-protocol.h"
#include "relative-pointer-unstable-v1-client-protocol.h"
#include "tablet-v2-client-protocol.h"
#include "text-input-unstable-v3-client-protocol.h"
#include "viewporter-client-protocol.h"
#include "xdg-activation-v1-client-protocol.h"
#include "xdg-decoration-unstable-v1-client-protocol.h"
#include "xdg-dialog-v1-client-protocol.h"
#include "xdg-foreign-unstable-v2-client-protocol.h"
#include "xdg-output-unstable-v1-client-protocol.h"
#include "xdg-shell-client-protocol.h"
#include "xdg-toplevel-icon-v1-client-protocol.h"
#include "color-management-v1-client-protocol.h"

#ifdef HAVE_LIBDECOR_H
#include <libdecor.h>
#endif

#define WAYLANDVID_DRIVER_NAME "wayland"

// Clamp certain core protocol versions on older versions of libwayland.
#if SDL_WAYLAND_CHECK_VERSION(1, 22, 0)
#define SDL_WL_COMPOSITOR_VERSION 6
#else
#define SDL_WL_COMPOSITOR_VERSION 4
#endif

#if SDL_WAYLAND_CHECK_VERSION(1, 22, 0)
#define SDL_WL_SEAT_VERSION 9
#elif SDL_WAYLAND_CHECK_VERSION(1, 21, 0)
#define SDL_WL_SEAT_VERSION 8
#else
#define SDL_WL_SEAT_VERSION 5
#endif

#if SDL_WAYLAND_CHECK_VERSION(1, 20, 0)
#define SDL_WL_OUTPUT_VERSION 4
#else
#define SDL_WL_OUTPUT_VERSION 3
#endif

#ifdef SDL_USE_LIBDBUS
#include "../../core/linux/SDL_dbus.h"

#define DISPLAY_INFO_NODE   "org.gnome.Mutter.DisplayConfig"
#define DISPLAY_INFO_PATH   "/org/gnome/Mutter/DisplayConfig"
#define DISPLAY_INFO_METHOD "GetCurrentState"
#endif

/* GNOME doesn't expose displays in any particular order, but we can find the
 * primary display and its logical coordinates via a DBus method.
 */
static bool Wayland_GetGNOMEPrimaryDisplayCoordinates(int *x, int *y)
{
#ifdef SDL_USE_LIBDBUS
    SDL_DBusContext *dbus = SDL_DBus_GetContext();
    if (dbus == NULL) {
        return false;
    }
    DBusMessage *reply = NULL;
    DBusMessageIter iter[3];
    DBusMessage *msg = dbus->message_new_method_call(DISPLAY_INFO_NODE,
                                                     DISPLAY_INFO_PATH,
                                                     DISPLAY_INFO_NODE,
                                                     DISPLAY_INFO_METHOD);

    if (msg) {
        reply = dbus->connection_send_with_reply_and_block(dbus->session_conn, msg, DBUS_TIMEOUT_USE_DEFAULT, NULL);
        dbus->message_unref(msg);
    }

    if (reply) {
        // Serial (don't care)
        dbus->message_iter_init(reply, &iter[0]);
        if (dbus->message_iter_get_arg_type(&iter[0]) != DBUS_TYPE_UINT32) {
            goto error;
        }

        // Physical monitor array (don't care)
        dbus->message_iter_next(&iter[0]);
        if (dbus->message_iter_get_arg_type(&iter[0]) != DBUS_TYPE_ARRAY) {
            goto error;
        }

        // Logical monitor array of structs
        dbus->message_iter_next(&iter[0]);
        if (dbus->message_iter_get_arg_type(&iter[0]) != DBUS_TYPE_ARRAY) {
            goto error;
        }

        // First logical monitor struct
        dbus->message_iter_recurse(&iter[0], &iter[1]);
        if (dbus->message_iter_get_arg_type(&iter[1]) != DBUS_TYPE_STRUCT) {
            goto error;
        }

        do {
            int logical_x, logical_y;
            dbus_bool_t primary;

            // Logical X
            dbus->message_iter_recurse(&iter[1], &iter[2]);
            if (dbus->message_iter_get_arg_type(&iter[2]) != DBUS_TYPE_INT32) {
                goto error;
            }
            dbus->message_iter_get_basic(&iter[2], &logical_x);

            // Logical Y
            dbus->message_iter_next(&iter[2]);
            if (dbus->message_iter_get_arg_type(&iter[2]) != DBUS_TYPE_INT32) {
                goto error;
            }
            dbus->message_iter_get_basic(&iter[2], &logical_y);

            // Scale (don't care)
            dbus->message_iter_next(&iter[2]);
            if (dbus->message_iter_get_arg_type(&iter[2]) != DBUS_TYPE_DOUBLE) {
                goto error;
            }

            // Transform (don't care)
            dbus->message_iter_next(&iter[2]);
            if (dbus->message_iter_get_arg_type(&iter[2]) != DBUS_TYPE_UINT32) {
                goto error;
            }

            // Primary display boolean
            dbus->message_iter_next(&iter[2]);
            if (dbus->message_iter_get_arg_type(&iter[2]) != DBUS_TYPE_BOOLEAN) {
                goto error;
            }
            dbus->message_iter_get_basic(&iter[2], &primary);

            if (primary) {
                *x = logical_x;
                *y = logical_y;

                // We found the primary display: success.
                dbus->message_unref(reply);
                return true;
            }
        } while (dbus->message_iter_next(&iter[1]));
    }

error:
    if (reply) {
        dbus->message_unref(reply);
    }
#endif
    return false;
}

// Sort the list of displays into a deterministic order
static int SDLCALL Wayland_DisplayPositionCompare(const void *a, const void *b)
{
    const SDL_DisplayData *da = *(SDL_DisplayData **)a;
    const SDL_DisplayData *db = *(SDL_DisplayData **)b;

    const bool a_at_origin = da->x == 0 && da->y == 0;
    const bool b_at_origin = db->x == 0 && db->y == 0;

    // Sort the display at 0,0 to be beginning of the list, as that will be the fallback primary.
    if (a_at_origin && !b_at_origin) {
        return -1;
    }
    if (b_at_origin && !a_at_origin) {
        return 1;
    }
    if (da->x < db->x) {
        return -1;
    }
    if (da->x > db->x) {
        return 1;
    }
    if (da->y < db->y) {
        return -1;
    }
    if (da->y > db->y) {
        return 1;
    }

    // If no position information is available, use the connector name.
    if (da->wl_output_name && db->wl_output_name) {
        return SDL_strcmp(da->wl_output_name, db->wl_output_name);
    }

    return 0;
}

/* Wayland doesn't have the native concept of a primary display, but there are clients that
 * will base their resolution lists on, or automatically make themselves fullscreen on, the
 * first listed output, which can lead to problems if the first listed output isn't
 * necessarily the best display for this. This attempts to find a primary display, first by
 * querying the GNOME DBus property, then trying to determine the 'best' display if that fails.
 * If all displays are equal, the one at position 0,0 will become the primary.
 *
 * The primary is determined by the following criteria, in order:
 * - Landscape is preferred over portrait
 * - The highest native resolution
 * - A higher HDR range is preferred
 * - Higher refresh is preferred (ignoring small differences)
 * - Lower scale values are preferred (larger display)
 */
static int Wayland_GetPrimaryDisplay(SDL_VideoData *vid)
{
    static const int REFRESH_DELTA = 4000;

    // Query the DBus interface to see if the coordinates of the primary display are exposed.
    int x, y;
    if (Wayland_GetGNOMEPrimaryDisplayCoordinates(&x, &y)) {
        for (int i = 0; i < vid->output_count; ++i) {
            if (vid->output_list[i]->x == x && vid->output_list[i]->y == y) {
                return i;
            }
        }
    }

    // Otherwise, choose the 'best' display.
    int best_width = 0;
    int best_height = 0;
    double best_scale = 0.0;
    float best_headroom = 0.0f;
    int best_refresh = 0;
    bool best_is_landscape = false;
    int best_index = 0;

    for (int i = 0; i < vid->output_count; ++i) {
        const SDL_DisplayData *d = vid->output_list[i];
        const bool is_landscape = d->orientation != SDL_ORIENTATION_PORTRAIT && d->orientation != SDL_ORIENTATION_PORTRAIT_FLIPPED;
        bool have_new_best = false;

        if (!best_is_landscape && is_landscape) { // Favor landscape over portrait displays.
            have_new_best = true;
        } else if (!best_is_landscape || is_landscape) { // Ignore portrait displays if a landscape was already found.
            if (d->pixel_width > best_width || d->pixel_height > best_height) {
                have_new_best = true;
            } else if (d->pixel_width == best_width && d->pixel_height == best_height) {
                if (d->HDR.HDR_headroom > best_headroom) { // Favor a higher HDR luminance range
                    have_new_best = true;
                } else if (d->HDR.HDR_headroom == best_headroom) {
                    if (d->refresh - best_refresh > REFRESH_DELTA) { // Favor a higher refresh rate, but ignore small differences (e.g. 59.97 vs 60.1)
                        have_new_best = true;
                    } else if (d->scale_factor < best_scale && SDL_abs(d->refresh - best_refresh) <= REFRESH_DELTA) {
                        // Prefer a lower scale display if the difference in refresh rate is small.
                        have_new_best = true;
                    }
                }
            }
        }

        if (have_new_best) {
            best_width = d->pixel_width;
            best_height = d->pixel_height;
            best_scale = d->scale_factor;
            best_headroom = d->HDR.HDR_headroom;
            best_refresh = d->refresh;
            best_is_landscape = is_landscape;
            best_index = i;
        }
    }

    return best_index;
}

static void Wayland_SortOutputsByPriorityHint(SDL_VideoData *vid)
{
    const char *name_hint = SDL_GetHint(SDL_HINT_VIDEO_DISPLAY_PRIORITY);

    if (name_hint) {
        char *saveptr;
        char *str = SDL_strdup(name_hint);
        SDL_DisplayData **sorted_list = SDL_malloc(sizeof(SDL_DisplayData *) * vid->output_count);

        if (str && sorted_list) {
            int sorted_index = 0;

            // Sort the requested displays to the front of the list.
            const char *token = SDL_strtok_r(str, ",", &saveptr);
            while (token) {
                for (int i = 0; i < vid->output_count; ++i) {
                    SDL_DisplayData *d = vid->output_list[i];
                    if (d && d->wl_output_name && SDL_strcmp(token, d->wl_output_name) == 0) {
                        sorted_list[sorted_index++] = d;
                        vid->output_list[i] = NULL;
                        break;
                    }
                }

                token = SDL_strtok_r(NULL, ",", &saveptr);
            }

            // Append the remaining outputs to the end of the list.
            for (int i = 0; i < vid->output_count; ++i) {
                if (vid->output_list[i]) {
                    sorted_list[sorted_index++] = vid->output_list[i];
                }
            }

            // Copy the sorted list to the output list.
            SDL_memcpy(vid->output_list, sorted_list, sizeof(SDL_DisplayData *) * vid->output_count);
        }

        SDL_free(str);
        SDL_free(sorted_list);
    }
}

static void Wayland_SortOutputs(SDL_VideoData *vid)
{
    // Sort by position or connector name, so the order of outputs is deterministic.
    SDL_qsort(vid->output_list, vid->output_count, sizeof(SDL_DisplayData *), Wayland_DisplayPositionCompare);

    // Find a suitable primary display and move it to the front of the list.
    const int primary_index = Wayland_GetPrimaryDisplay(vid);
    if (primary_index) {
        SDL_DisplayData *primary = vid->output_list[primary_index];
        SDL_memmove(&vid->output_list[1], &vid->output_list[0], sizeof(SDL_DisplayData *) * primary_index);
        vid->output_list[0] = primary;
    }

    // Apply the ordering hint, if specified.
    Wayland_SortOutputsByPriorityHint(vid);
}

static void display_handle_done(void *data, struct wl_output *output);

// Initialization/Query functions
static bool Wayland_VideoInit(SDL_VideoDevice *_this);
static bool Wayland_GetDisplayBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect);
static void Wayland_VideoQuit(SDL_VideoDevice *_this);

static const char *SDL_WAYLAND_surface_tag = "sdl-window";
static const char *SDL_WAYLAND_output_tag = "sdl-output";

void SDL_WAYLAND_register_surface(struct wl_surface *surface)
{
    wl_proxy_set_tag((struct wl_proxy *)surface, &SDL_WAYLAND_surface_tag);
}

void SDL_WAYLAND_register_output(struct wl_output *output)
{
    wl_proxy_set_tag((struct wl_proxy *)output, &SDL_WAYLAND_output_tag);
}

bool SDL_WAYLAND_own_surface(struct wl_surface *surface)
{
    return wl_proxy_get_tag((struct wl_proxy *)surface) == &SDL_WAYLAND_surface_tag;
}

bool SDL_WAYLAND_own_output(struct wl_output *output)
{
    return wl_proxy_get_tag((struct wl_proxy *)output) == &SDL_WAYLAND_output_tag;
}

/* External surfaces may have their own user data attached, the modification of which
 * can cause problems with external toolkits. Instead, external windows are kept in
 * their own list, and a search is conducted to find a matching surface.
 */
static struct wl_list external_window_list;

void Wayland_AddWindowDataToExternalList(SDL_WindowData *data)
{
    WAYLAND_wl_list_insert(&external_window_list, &data->external_window_list_link);
}

void Wayland_RemoveWindowDataFromExternalList(SDL_WindowData *data)
{
    WAYLAND_wl_list_remove(&data->external_window_list_link);
}

SDL_WindowData *Wayland_GetWindowDataForOwnedSurface(struct wl_surface *surface)
{
    if (SDL_WAYLAND_own_surface(surface)) {
        return (SDL_WindowData *)wl_surface_get_user_data(surface);
    } else if (!WAYLAND_wl_list_empty(&external_window_list)) {
        SDL_WindowData *p;
        wl_list_for_each (p, &external_window_list, external_window_list_link) {
            if (p->surface == surface) {
                return p;
            }
        }
    }

    return NULL;
}

static void Wayland_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_VideoData *data = device->internal;
    if (data->display && !data->display_externally_owned) {
        WAYLAND_wl_display_flush(data->display);
        WAYLAND_wl_display_disconnect(data->display);
        SDL_ClearProperty(SDL_GetGlobalProperties(), SDL_PROP_GLOBAL_VIDEO_WAYLAND_WL_DISPLAY_POINTER);
    }
    if (device->wakeup_lock) {
        SDL_DestroyMutex(device->wakeup_lock);
    }
    SDL_free(data);
    SDL_free(device);
    SDL_WAYLAND_UnloadSymbols();
}

typedef struct
{
    bool has_fifo_v1;
} SDL_WaylandPreferredData;

static void wayland_preferred_check_handle_global(void *data, struct wl_registry *registry, uint32_t id,
                                                  const char *interface, uint32_t version)
{
    SDL_WaylandPreferredData *d = data;

    if (SDL_strcmp(interface, "wp_fifo_manager_v1") == 0) {
        d->has_fifo_v1 = true;
    }
}

static void wayland_preferred_check_remove_global(void *data, struct wl_registry *registry, uint32_t id)
{
    // No need to do anything here.
}

static const struct wl_registry_listener preferred_registry_listener = {
    wayland_preferred_check_handle_global,
    wayland_preferred_check_remove_global
};

static bool Wayland_IsPreferred(struct wl_display *display)
{
    struct wl_registry *registry = wl_display_get_registry(display);
    SDL_WaylandPreferredData preferred_data = { 0 };

    if (!registry) {
        SDL_SetError("Failed to get the Wayland registry");
        return false;
    }

    wl_registry_add_listener(registry, &preferred_registry_listener, &preferred_data);

    WAYLAND_wl_display_roundtrip(display);

    wl_registry_destroy(registry);

    if (!preferred_data.has_fifo_v1) {
        SDL_LogInfo(SDL_LOG_CATEGORY_VIDEO, "This compositor lacks support for the fifo-v1 protocol; falling back to XWayland for GPU performance reasons (set SDL_VIDEO_DRIVER=wayland to override)");
    }
    return preferred_data.has_fifo_v1;
}

static SDL_VideoDevice *Wayland_CreateDevice(bool require_preferred_protocols)
{
    SDL_VideoDevice *device;
    SDL_VideoData *data;
    struct wl_display *display = SDL_GetPointerProperty(SDL_GetGlobalProperties(),
                                                 SDL_PROP_GLOBAL_VIDEO_WAYLAND_WL_DISPLAY_POINTER, NULL);
    bool display_is_external = !!display;

    // Are we trying to connect to or are currently in a Wayland session?
    if (!SDL_getenv("WAYLAND_DISPLAY")) {
        const char *session = SDL_getenv("XDG_SESSION_TYPE");
        if (session && SDL_strcasecmp(session, "wayland") != 0) {
            return NULL;
        }
    }

    if (!SDL_WAYLAND_LoadSymbols()) {
        return NULL;
    }

    if (!display) {
        display = WAYLAND_wl_display_connect(NULL);
        if (!display) {
            SDL_WAYLAND_UnloadSymbols();
            return NULL;
        }
    }

    /*
     * If we are checking for preferred Wayland, then let's query for
     * fifo-v1's existence, so we don't regress GPU-bound performance
     * and frame-pacing by default due to swapchain starvation.
     */
    if (require_preferred_protocols && !Wayland_IsPreferred(display)) {
        if (!display_is_external) {
            WAYLAND_wl_display_disconnect(display);
        }
        SDL_WAYLAND_UnloadSymbols();
        return NULL;
    }

    data = SDL_calloc(1, sizeof(*data));
    if (!data) {
        if (!display_is_external) {
            WAYLAND_wl_display_disconnect(display);
        }
        SDL_WAYLAND_UnloadSymbols();
        return NULL;
    }

    data->initializing = true;
    data->display = display;
    data->display_externally_owned = display_is_external;
    data->scale_to_display_enabled = SDL_GetHintBoolean(SDL_HINT_VIDEO_WAYLAND_SCALE_TO_DISPLAY, false);
    WAYLAND_wl_list_init(&data->seat_list);
    WAYLAND_wl_list_init(&external_window_list);

    // Initialize all variables that we clean on shutdown
    device = SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        SDL_free(data);
        if (!display_is_external) {
            WAYLAND_wl_display_disconnect(display);
        }
        SDL_WAYLAND_UnloadSymbols();
        return NULL;
    }

    if (!display_is_external) {
        SDL_SetPointerProperty(SDL_GetGlobalProperties(),
                        SDL_PROP_GLOBAL_VIDEO_WAYLAND_WL_DISPLAY_POINTER, display);
    }

    device->internal = data;
    device->wakeup_lock = SDL_CreateMutex();

    // Set the function pointers
    device->VideoInit = Wayland_VideoInit;
    device->VideoQuit = Wayland_VideoQuit;
    device->GetDisplayBounds = Wayland_GetDisplayBounds;
    device->SuspendScreenSaver = Wayland_SuspendScreenSaver;

    device->PumpEvents = Wayland_PumpEvents;
    device->WaitEventTimeout = Wayland_WaitEventTimeout;
    device->SendWakeupEvent = Wayland_SendWakeupEvent;

#ifdef SDL_VIDEO_OPENGL_EGL
    device->GL_SwapWindow = Wayland_GLES_SwapWindow;
    device->GL_GetSwapInterval = Wayland_GLES_GetSwapInterval;
    device->GL_SetSwapInterval = Wayland_GLES_SetSwapInterval;
    device->GL_MakeCurrent = Wayland_GLES_MakeCurrent;
    device->GL_CreateContext = Wayland_GLES_CreateContext;
    device->GL_LoadLibrary = Wayland_GLES_LoadLibrary;
    device->GL_UnloadLibrary = Wayland_GLES_UnloadLibrary;
    device->GL_GetProcAddress = Wayland_GLES_GetProcAddress;
    device->GL_DestroyContext = Wayland_GLES_DestroyContext;
    device->GL_GetEGLSurface = Wayland_GLES_GetEGLSurface;
#endif

    device->CreateSDLWindow = Wayland_CreateWindow;
    device->ShowWindow = Wayland_ShowWindow;
    device->HideWindow = Wayland_HideWindow;
    device->RaiseWindow = Wayland_RaiseWindow;
    device->SetWindowFullscreen = Wayland_SetWindowFullscreen;
    device->MaximizeWindow = Wayland_MaximizeWindow;
    device->MinimizeWindow = Wayland_MinimizeWindow;
    device->SetWindowMouseRect = Wayland_SetWindowMouseRect;
    device->SetWindowMouseGrab = Wayland_SetWindowMouseGrab;
    device->SetWindowKeyboardGrab = Wayland_SetWindowKeyboardGrab;
    device->RestoreWindow = Wayland_RestoreWindow;
    device->SetWindowBordered = Wayland_SetWindowBordered;
    device->SetWindowResizable = Wayland_SetWindowResizable;
    device->SetWindowPosition = Wayland_SetWindowPosition;
    device->SetWindowSize = Wayland_SetWindowSize;
    device->SetWindowMinimumSize = Wayland_SetWindowMinimumSize;
    device->SetWindowMaximumSize = Wayland_SetWindowMaximumSize;
    device->SetWindowParent = Wayland_SetWindowParent;
    device->SetWindowModal = Wayland_SetWindowModal;
    device->SetWindowOpacity = Wayland_SetWindowOpacity;
    device->SetWindowTitle = Wayland_SetWindowTitle;
    device->SetWindowIcon = Wayland_SetWindowIcon;
    device->GetWindowSizeInPixels = Wayland_GetWindowSizeInPixels;
    device->GetWindowContentScale = Wayland_GetWindowContentScale;
    device->GetWindowICCProfile = Wayland_GetWindowICCProfile;
    device->GetDisplayForWindow = Wayland_GetDisplayForWindow;
    device->DestroyWindow = Wayland_DestroyWindow;
    device->SetWindowHitTest = Wayland_SetWindowHitTest;
    device->FlashWindow = Wayland_FlashWindow;
    device->HasScreenKeyboardSupport = Wayland_HasScreenKeyboardSupport;
    device->ShowWindowSystemMenu = Wayland_ShowWindowSystemMenu;
    device->SyncWindow = Wayland_SyncWindow;
    device->SetWindowFocusable = Wayland_SetWindowFocusable;

#ifdef SDL_USE_LIBDBUS
    if (SDL_SystemTheme_Init())
        device->system_theme = SDL_SystemTheme_Get();
#endif

    device->GetTextMimeTypes = Wayland_GetTextMimeTypes;
    device->SetClipboardData = Wayland_SetClipboardData;
    device->GetClipboardData = Wayland_GetClipboardData;
    device->HasClipboardData = Wayland_HasClipboardData;
    device->StartTextInput = Wayland_StartTextInput;
    device->StopTextInput = Wayland_StopTextInput;
    device->UpdateTextInputArea = Wayland_UpdateTextInputArea;

#ifdef SDL_VIDEO_VULKAN
    device->Vulkan_LoadLibrary = Wayland_Vulkan_LoadLibrary;
    device->Vulkan_UnloadLibrary = Wayland_Vulkan_UnloadLibrary;
    device->Vulkan_GetInstanceExtensions = Wayland_Vulkan_GetInstanceExtensions;
    device->Vulkan_CreateSurface = Wayland_Vulkan_CreateSurface;
    device->Vulkan_DestroySurface = Wayland_Vulkan_DestroySurface;
    device->Vulkan_GetPresentationSupport = Wayland_Vulkan_GetPresentationSupport;
#endif

    device->free = Wayland_DeleteDevice;

    device->device_caps = VIDEO_DEVICE_CAPS_MODE_SWITCHING_EMULATED |
                          VIDEO_DEVICE_CAPS_HAS_POPUP_WINDOW_SUPPORT |
                          VIDEO_DEVICE_CAPS_SENDS_FULLSCREEN_DIMENSIONS |
                          VIDEO_DEVICE_CAPS_SENDS_DISPLAY_CHANGES |
                          VIDEO_DEVICE_CAPS_DISABLE_MOUSE_WARP_ON_FULLSCREEN_TRANSITIONS |
                          VIDEO_DEVICE_CAPS_SENDS_HDR_CHANGES;

    return device;
}

static SDL_VideoDevice *Wayland_Preferred_CreateDevice(void)
{
    return Wayland_CreateDevice(true);
}

static SDL_VideoDevice *Wayland_Fallback_CreateDevice(void)
{
    return Wayland_CreateDevice(false);
}

VideoBootStrap Wayland_preferred_bootstrap = {
    WAYLANDVID_DRIVER_NAME, "SDL Wayland video driver",
    Wayland_Preferred_CreateDevice,
    Wayland_ShowMessageBox,
    true
};

VideoBootStrap Wayland_bootstrap = {
    WAYLANDVID_DRIVER_NAME, "SDL Wayland video driver",
    Wayland_Fallback_CreateDevice,
    Wayland_ShowMessageBox,
    false
};

static void xdg_output_handle_logical_position(void *data, struct zxdg_output_v1 *xdg_output,
                                               int32_t x, int32_t y)
{
    SDL_DisplayData *internal = (SDL_DisplayData *)data;

    internal->x = x;
    internal->y = y;
    internal->has_logical_position = true;
}

static void xdg_output_handle_logical_size(void *data, struct zxdg_output_v1 *xdg_output,
                                           int32_t width, int32_t height)
{
    SDL_DisplayData *internal = (SDL_DisplayData *)data;

    internal->logical_width = width;
    internal->logical_height = height;
    internal->has_logical_size = true;
}

static void xdg_output_handle_done(void *data, struct zxdg_output_v1 *xdg_output)
{
    SDL_DisplayData *internal = data;

    /*
     * xdg-output.done events are deprecated and only apply below version 3 of the protocol.
     * A wl-output.done event will be emitted in version 3 or higher.
     */
    if (zxdg_output_v1_get_version(internal->xdg_output) < 3) {
        display_handle_done(data, internal->output);
    }
}

static void xdg_output_handle_name(void *data, struct zxdg_output_v1 *xdg_output,
                                   const char *name)
{
    SDL_DisplayData *internal = (SDL_DisplayData *)data;

    // Deprecated as of wl_output v4.
    if (wl_output_get_version(internal->output) < WL_OUTPUT_NAME_SINCE_VERSION &&
        internal->display == 0) {
        SDL_free(internal->wl_output_name);
        internal->wl_output_name = SDL_strdup(name);
    }
}

static void xdg_output_handle_description(void *data, struct zxdg_output_v1 *xdg_output,
                                          const char *description)
{
    SDL_DisplayData *internal = (SDL_DisplayData *)data;

    // Deprecated as of wl_output v4.
    if (wl_output_get_version(internal->output) < WL_OUTPUT_DESCRIPTION_SINCE_VERSION &&
        internal->display == 0) {
        // xdg-output descriptions, if available, supersede wl-output model names.
        SDL_free(internal->placeholder.name);
        internal->placeholder.name = SDL_strdup(description);
    }
}

static const struct zxdg_output_v1_listener xdg_output_listener = {
    xdg_output_handle_logical_position,
    xdg_output_handle_logical_size,
    xdg_output_handle_done,
    xdg_output_handle_name,
    xdg_output_handle_description,
};

static void AddEmulatedModes(SDL_DisplayData *dispdata, int native_width, int native_height)
{
    struct EmulatedMode
    {
        int w;
        int h;
    };

    // Resolution lists courtesy of XWayland
    const struct EmulatedMode mode_list[] = {
        // 16:9 (1.77)
        { 7680, 4320 },
        { 6144, 3160 },
        { 5120, 2880 },
        { 4096, 2304 },
        { 3840, 2160 },
        { 3200, 1800 },
        { 2880, 1620 },
        { 2560, 1440 },
        { 2048, 1152 },
        { 1920, 1080 },
        { 1600, 900 },
        { 1368, 768 },
        { 1280, 720 },
        { 864, 486 },

        // 16:10 (1.6)
        { 2560, 1600 },
        { 1920, 1200 },
        { 1680, 1050 },
        { 1440, 900 },
        { 1280, 800 },

        // 3:2 (1.5)
        { 720, 480 },

        // 4:3 (1.33)
        { 2048, 1536 },
        { 1920, 1440 },
        { 1600, 1200 },
        { 1440, 1080 },
        { 1400, 1050 },
        { 1280, 1024 },
        { 1280, 960 },
        { 1152, 864 },
        { 1024, 768 },
        { 800, 600 },
        { 640, 480 }
    };

    int i;
    SDL_DisplayMode mode;
    SDL_VideoDisplay *dpy = dispdata->display ? SDL_GetVideoDisplay(dispdata->display) : &dispdata->placeholder;
    const bool rot_90 = native_width < native_height; // Reverse width/height for portrait displays.

    for (i = 0; i < SDL_arraysize(mode_list); ++i) {
        SDL_zero(mode);
        mode.format = dpy->desktop_mode.format;
        mode.refresh_rate_numerator = dpy->desktop_mode.refresh_rate_numerator;
        mode.refresh_rate_denominator = dpy->desktop_mode.refresh_rate_denominator;

        if (rot_90) {
            mode.w = mode_list[i].h;
            mode.h = mode_list[i].w;
        } else {
            mode.w = mode_list[i].w;
            mode.h = mode_list[i].h;
        }

        // Only add modes that are smaller than the native mode.
        if ((mode.w < native_width && mode.h < native_height) ||
            (mode.w < native_width && mode.h == native_height) ||
            (mode.w == native_width && mode.h < native_height)) {
            SDL_AddFullscreenDisplayMode(dpy, &mode);
        }
    }
}

static void display_handle_geometry(void *data,
                                    struct wl_output *output,
                                    int x, int y,
                                    int physical_width,
                                    int physical_height,
                                    int subpixel,
                                    const char *make,
                                    const char *model,
                                    int transform)

{
    SDL_DisplayData *internal = (SDL_DisplayData *)data;

    // Apply the change from wl-output only if xdg-output is not supported
    if (!internal->has_logical_position) {
        internal->x = x;
        internal->y = y;
    }
    internal->physical_width_mm = physical_width;
    internal->physical_height_mm = physical_height;

    // The model is only used for the output name if wl_output or xdg-output haven't provided a description.
    if (internal->display == 0 && !internal->placeholder.name) {
        internal->placeholder.name = SDL_strdup(model);
    }

    internal->transform = transform;
#define TF_CASE(in, out)                                 \
    case WL_OUTPUT_TRANSFORM_##in:                       \
        internal->orientation = SDL_ORIENTATION_##out; \
        break;
    if (internal->physical_width_mm >= internal->physical_height_mm) {
        switch (transform) {
            TF_CASE(NORMAL, LANDSCAPE)
            TF_CASE(90, PORTRAIT)
            TF_CASE(180, LANDSCAPE_FLIPPED)
            TF_CASE(270, PORTRAIT_FLIPPED)
            TF_CASE(FLIPPED, LANDSCAPE_FLIPPED)
            TF_CASE(FLIPPED_90, PORTRAIT_FLIPPED)
            TF_CASE(FLIPPED_180, LANDSCAPE)
            TF_CASE(FLIPPED_270, PORTRAIT)
        }
    } else {
        switch (transform) {
            TF_CASE(NORMAL, PORTRAIT)
            TF_CASE(90, LANDSCAPE)
            TF_CASE(180, PORTRAIT_FLIPPED)
            TF_CASE(270, LANDSCAPE_FLIPPED)
            TF_CASE(FLIPPED, PORTRAIT_FLIPPED)
            TF_CASE(FLIPPED_90, LANDSCAPE_FLIPPED)
            TF_CASE(FLIPPED_180, PORTRAIT)
            TF_CASE(FLIPPED_270, LANDSCAPE)
        }
    }
#undef TF_CASE
}

static void display_handle_mode(void *data,
                                struct wl_output *output,
                                uint32_t flags,
                                int width,
                                int height,
                                int refresh)
{
    SDL_DisplayData *internal = (SDL_DisplayData *)data;

    if (flags & WL_OUTPUT_MODE_CURRENT) {
        internal->pixel_width = width;
        internal->pixel_height = height;

        /*
         * Don't rotate this yet, wl-output coordinates are transformed in
         * handle_done and xdg-output coordinates are pre-transformed.
         */
        if (!internal->has_logical_size) {
            internal->logical_width = width;
            internal->logical_height = height;
        }

        internal->refresh = refresh;
    }
}

static void display_handle_done(void *data,
                                struct wl_output *output)
{
    const bool mode_emulation_enabled = SDL_GetHintBoolean(SDL_HINT_VIDEO_WAYLAND_MODE_EMULATION, true);
    SDL_DisplayData *internal = (SDL_DisplayData *)data;
    SDL_VideoData *video = internal->videodata;
    SDL_DisplayMode native_mode, desktop_mode;

    /*
     * When using xdg-output, two wl-output.done events will be emitted:
     * one at the completion of wl-display and one at the completion of xdg-output.
     *
     * All required events must be received before proceeding.
     */
    const int event_await_count = 1 + (internal->xdg_output != NULL);

    internal->wl_output_done_count = SDL_min(internal->wl_output_done_count + 1, event_await_count + 1);

    if (internal->wl_output_done_count < event_await_count) {
        return;
    }

    // If the display was already created, reset and rebuild the mode list.
    SDL_VideoDisplay *dpy = SDL_GetVideoDisplay(internal->display);
    if (dpy) {
        SDL_ResetFullscreenDisplayModes(dpy);
    }

    // The native display resolution
    SDL_zero(native_mode);
    native_mode.format = SDL_PIXELFORMAT_XRGB8888;

    // Transform the pixel values, if necessary.
    if (internal->transform & WL_OUTPUT_TRANSFORM_90) {
        native_mode.w = internal->pixel_height;
        native_mode.h = internal->pixel_width;
    } else {
        native_mode.w = internal->pixel_width;
        native_mode.h = internal->pixel_height;
    }
    native_mode.refresh_rate_numerator = internal->refresh;
    native_mode.refresh_rate_denominator = 1000;

    if (internal->has_logical_size) { // If xdg-output is present...
        if (native_mode.w != internal->logical_width || native_mode.h != internal->logical_height) {
            // ...and the compositor scales the logical viewport...
            if (video->viewporter) {
                // ...and viewports are supported, calculate the true scale of the output.
                internal->scale_factor = (double)native_mode.w / (double)internal->logical_width;
            } else {
                // ...otherwise, the 'native' pixel values are a multiple of the logical screen size.
                internal->pixel_width = internal->logical_width * (int)internal->scale_factor;
                internal->pixel_height = internal->logical_height * (int)internal->scale_factor;
            }
        } else {
            /* ...and the output viewport is not scaled in the global compositing
             * space, the output dimensions need to be divided by the scale factor.
             */
            internal->logical_width /= (int)internal->scale_factor;
            internal->logical_height /= (int)internal->scale_factor;
        }
    } else {
        /* Calculate the points from the pixel values, if xdg-output isn't present.
         * Use the native mode pixel values since they are pre-transformed.
         */
        internal->logical_width = native_mode.w / (int)internal->scale_factor;
        internal->logical_height = native_mode.h / (int)internal->scale_factor;
    }

    // The scaled desktop mode
    SDL_zero(desktop_mode);
    desktop_mode.format = SDL_PIXELFORMAT_XRGB8888;

    if (!video->scale_to_display_enabled) {
        desktop_mode.w = internal->logical_width;
        desktop_mode.h = internal->logical_height;
        desktop_mode.pixel_density = (float)internal->scale_factor;
    } else {
        desktop_mode.w = native_mode.w;
        desktop_mode.h = native_mode.h;
        desktop_mode.pixel_density = 1.0f;
    }

    desktop_mode.refresh_rate_numerator = internal->refresh;
    desktop_mode.refresh_rate_denominator = 1000;

    if (internal->display > 0) {
        dpy = SDL_GetVideoDisplay(internal->display);
    } else {
        dpy = &internal->placeholder;
    }

    if (video->scale_to_display_enabled) {
        SDL_SetDisplayContentScale(dpy, (float)internal->scale_factor);
    }

    // Set the desktop display mode.
    SDL_SetDesktopDisplayMode(dpy, &desktop_mode);

    // Expose the unscaled, native resolution if the scale is 1.0 or viewports are available...
    if (internal->scale_factor == 1.0 || video->viewporter) {
        SDL_AddFullscreenDisplayMode(dpy, &native_mode);
        if (native_mode.w != desktop_mode.w ||
            native_mode.h != desktop_mode.h) {
            SDL_AddFullscreenDisplayMode(dpy, &desktop_mode);
        }
    } else {
        // ...otherwise expose the integer scaled variants of the desktop resolution down to 1.
        int i;

        desktop_mode.pixel_density = 1.0f;

        for (i = (int)internal->scale_factor; i > 0; --i) {
            desktop_mode.w = internal->logical_width * i;
            desktop_mode.h = internal->logical_height * i;
            SDL_AddFullscreenDisplayMode(dpy, &desktop_mode);
        }
    }

    // Add emulated modes if wp_viewporter is supported and mode emulation is enabled.
    if (video->viewporter && mode_emulation_enabled) {
        // The transformed display pixel width/height must be used here.
        AddEmulatedModes(internal, native_mode.w, native_mode.h);
    }

    SDL_SetDisplayHDRProperties(dpy, &internal->HDR);

    if (internal->display == 0) {
        // First time getting display info, initialize the VideoDisplay
        if (internal->physical_width_mm >= internal->physical_height_mm) {
            internal->placeholder.natural_orientation = SDL_ORIENTATION_LANDSCAPE;
        } else {
            internal->placeholder.natural_orientation = SDL_ORIENTATION_PORTRAIT;
        }
        internal->placeholder.current_orientation = internal->orientation;
        internal->placeholder.internal = internal;

        internal->placeholder.props = SDL_CreateProperties();
        SDL_SetPointerProperty(internal->placeholder.props, SDL_PROP_DISPLAY_WAYLAND_WL_OUTPUT_POINTER, internal->output);

        // During initialization, the displays will be added after enumeration is complete.
        if (!video->initializing) {
            internal->display = SDL_AddVideoDisplay(&internal->placeholder, true);
            SDL_free(internal->placeholder.name);
            SDL_zero(internal->placeholder);
        }
    } else {
        SDL_SendDisplayEvent(dpy, SDL_EVENT_DISPLAY_ORIENTATION, internal->orientation, 0);
    }
}

static void display_handle_scale(void *data,
                                 struct wl_output *output,
                                 int32_t factor)
{
    SDL_DisplayData *internal = (SDL_DisplayData *)data;
    internal->scale_factor = factor;
}

static void display_handle_name(void *data, struct wl_output *wl_output, const char *name)
{
    SDL_DisplayData *internal = (SDL_DisplayData *)data;

    SDL_free(internal->wl_output_name);
    internal->wl_output_name = SDL_strdup(name);
}

static void display_handle_description(void *data, struct wl_output *wl_output, const char *description)
{
    SDL_DisplayData *internal = (SDL_DisplayData *)data;

    if (internal->display == 0) {
        // The description, if available, supersedes the model name.
        SDL_free(internal->placeholder.name);
        internal->placeholder.name = SDL_strdup(description);
    }
}

static const struct wl_output_listener output_listener = {
    display_handle_geometry,   // Version 1
    display_handle_mode,       // Version 1
    display_handle_done,       // Version 2
    display_handle_scale,      // Version 2
    display_handle_name,       // Version 4
    display_handle_description // Version 4
};

static void handle_output_image_description_changed(void *data,
                                                    struct wp_color_management_output_v1 *wp_color_management_output_v1)
{
    SDL_DisplayData *display = (SDL_DisplayData *)data;
    // wl_display.done is called after this event, so the display HDR status will be updated there.
    Wayland_GetColorInfoForOutput(display, false);
}

static const struct wp_color_management_output_v1_listener wp_color_management_output_listener = {
    handle_output_image_description_changed
};

static bool Wayland_add_display(SDL_VideoData *d, uint32_t id, uint32_t version)
{
    struct wl_output *output;
    SDL_DisplayData *data;

    output = wl_registry_bind(d->registry, id, &wl_output_interface, version);
    if (!output) {
        return SDL_SetError("Failed to retrieve output.");
    }
    data = (SDL_DisplayData *)SDL_calloc(1, sizeof(*data));
    data->videodata = d;
    data->output = output;
    data->registry_id = id;
    data->scale_factor = 1.0f;

    wl_output_add_listener(output, &output_listener, data);
    SDL_WAYLAND_register_output(output);

    // Keep a list of outputs for sorting and deferred protocol initialization.
    if (d->output_count == d->output_max) {
        d->output_max += 4;
        d->output_list = SDL_realloc(d->output_list, sizeof(SDL_DisplayData *) * d->output_max);
    }
    d->output_list[d->output_count++] = data;

    if (data->videodata->xdg_output_manager) {
        data->xdg_output = zxdg_output_manager_v1_get_xdg_output(data->videodata->xdg_output_manager, output);
        zxdg_output_v1_add_listener(data->xdg_output, &xdg_output_listener, data);
    }
    if (data->videodata->wp_color_manager_v1) {
        data->wp_color_management_output = wp_color_manager_v1_get_output(data->videodata->wp_color_manager_v1, output);
        wp_color_management_output_v1_add_listener(data->wp_color_management_output, &wp_color_management_output_listener, data);
        Wayland_GetColorInfoForOutput(data, true);
    }
    return true;
}

static void Wayland_free_display(SDL_VideoDisplay *display, bool send_event)
{
    if (display) {
        SDL_DisplayData *display_data = display->internal;

        /* A preceding surface leave event is not guaranteed when an output is removed,
         * so ensure that no window continues to hold a reference to a removed output.
         */
        for (SDL_Window *window = SDL_GetVideoDevice()->windows; window; window = window->next) {
            Wayland_RemoveOutputFromWindow(window->internal, display_data);
        }

        SDL_free(display_data->wl_output_name);

        if (display_data->wp_color_management_output) {
            Wayland_FreeColorInfoState(display_data->color_info_state);
            wp_color_management_output_v1_destroy(display_data->wp_color_management_output);
        }

        if (display_data->xdg_output) {
            zxdg_output_v1_destroy(display_data->xdg_output);
        }

        if (wl_output_get_version(display_data->output) >= WL_OUTPUT_RELEASE_SINCE_VERSION) {
            wl_output_release(display_data->output);
        } else {
            wl_output_destroy(display_data->output);
        }

        SDL_DelVideoDisplay(display->id, send_event);
    }
}

static void Wayland_FinalizeDisplays(SDL_VideoData *vid)
{
    Wayland_SortOutputs(vid);
    for(int i = 0; i < vid->output_count; ++i) {
        SDL_DisplayData *d = vid->output_list[i];
        d->display = SDL_AddVideoDisplay(&d->placeholder, false);
        SDL_free(d->placeholder.name);
        SDL_zero(d->placeholder);
    }
}

static void Wayland_init_xdg_output(SDL_VideoData *d)
{
    for (int i = 0; i < d->output_count; ++i) {
        SDL_DisplayData *disp = d->output_list[i];
        disp->xdg_output = zxdg_output_manager_v1_get_xdg_output(disp->videodata->xdg_output_manager, disp->output);
        zxdg_output_v1_add_listener(disp->xdg_output, &xdg_output_listener, disp);
    }
}

static void Wayland_InitColorManager(SDL_VideoData *d)
{
    for (int i = 0; i < d->output_count; ++i) {
        SDL_DisplayData *disp = d->output_list[i];
        disp->wp_color_management_output = wp_color_manager_v1_get_output(disp->videodata->wp_color_manager_v1, disp->output);
        wp_color_management_output_v1_add_listener(disp->wp_color_management_output, &wp_color_management_output_listener, disp);
        Wayland_GetColorInfoForOutput(disp, true);
    }
}

static void handle_ping_xdg_wm_base(void *data, struct xdg_wm_base *xdg, uint32_t serial)
{
    xdg_wm_base_pong(xdg, serial);
}

static const struct xdg_wm_base_listener shell_listener_xdg = {
    handle_ping_xdg_wm_base
};

#ifdef HAVE_LIBDECOR_H
static void libdecor_error(struct libdecor *context,
                           enum libdecor_error error,
                           const char *message)
{
    SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "libdecor error (%d): %s", error, message);
}

static struct libdecor_interface libdecor_interface = {
    libdecor_error,
};
#endif

static void display_handle_global(void *data, struct wl_registry *registry, uint32_t id,
                                  const char *interface, uint32_t version)
{
    SDL_VideoData *d = data;

    // printf("WAYLAND INTERFACE: %s\n", interface);

    if (SDL_strcmp(interface, "wl_compositor") == 0) {
        d->compositor = wl_registry_bind(d->registry, id, &wl_compositor_interface, SDL_min(SDL_WL_COMPOSITOR_VERSION, version));
    } else if (SDL_strcmp(interface, "wl_output") == 0) {
        Wayland_add_display(d, id, SDL_min(version, SDL_WL_OUTPUT_VERSION));
    } else if (SDL_strcmp(interface, "wl_seat") == 0) {
        struct wl_seat *seat = wl_registry_bind(d->registry, id, &wl_seat_interface, SDL_min(SDL_WL_SEAT_VERSION, version));
        Wayland_DisplayCreateSeat(d, seat, id);
    } else if (SDL_strcmp(interface, "xdg_wm_base") == 0) {
        d->shell.xdg = wl_registry_bind(d->registry, id, &xdg_wm_base_interface, SDL_min(version, 7));
        xdg_wm_base_add_listener(d->shell.xdg, &shell_listener_xdg, NULL);
    } else if (SDL_strcmp(interface, "wl_shm") == 0) {
        d->shm = wl_registry_bind(registry, id, &wl_shm_interface, 1);
    } else if (SDL_strcmp(interface, "zwp_relative_pointer_manager_v1") == 0) {
        d->relative_pointer_manager = wl_registry_bind(d->registry, id, &zwp_relative_pointer_manager_v1_interface, 1);
        Wayland_DisplayInitRelativePointerManager(d);
    } else if (SDL_strcmp(interface, "zwp_pointer_constraints_v1") == 0) {
        d->pointer_constraints = wl_registry_bind(d->registry, id, &zwp_pointer_constraints_v1_interface, 1);
    } else if (SDL_strcmp(interface, "zwp_keyboard_shortcuts_inhibit_manager_v1") == 0) {
        d->key_inhibitor_manager = wl_registry_bind(d->registry, id, &zwp_keyboard_shortcuts_inhibit_manager_v1_interface, 1);
    } else if (SDL_strcmp(interface, "zwp_idle_inhibit_manager_v1") == 0) {
        d->idle_inhibit_manager = wl_registry_bind(d->registry, id, &zwp_idle_inhibit_manager_v1_interface, 1);
    } else if (SDL_strcmp(interface, "xdg_activation_v1") == 0) {
        d->activation_manager = wl_registry_bind(d->registry, id, &xdg_activation_v1_interface, 1);
    } else if (SDL_strcmp(interface, "zwp_text_input_manager_v3") == 0) {
        Wayland_DisplayCreateTextInputManager(d, id);
    } else if (SDL_strcmp(interface, "wl_data_device_manager") == 0) {
        d->data_device_manager = wl_registry_bind(d->registry, id, &wl_data_device_manager_interface, SDL_min(3, version));
        Wayland_DisplayInitDataDeviceManager(d);
    } else if (SDL_strcmp(interface, "zwp_primary_selection_device_manager_v1") == 0) {
        d->primary_selection_device_manager = wl_registry_bind(d->registry, id, &zwp_primary_selection_device_manager_v1_interface, 1);
        Wayland_DisplayInitPrimarySelectionDeviceManager(d);
    } else if (SDL_strcmp(interface, "zxdg_decoration_manager_v1") == 0) {
        d->decoration_manager = wl_registry_bind(d->registry, id, &zxdg_decoration_manager_v1_interface, 1);
    } else if (SDL_strcmp(interface, "zwp_tablet_manager_v2") == 0) {
        d->tablet_manager = wl_registry_bind(d->registry, id, &zwp_tablet_manager_v2_interface, 1);
        Wayland_DisplayInitTabletManager(d);
    } else if (SDL_strcmp(interface, "zxdg_output_manager_v1") == 0) {
        version = SDL_min(version, 3); // Versions 1 through 3 are supported.
        d->xdg_output_manager = wl_registry_bind(d->registry, id, &zxdg_output_manager_v1_interface, version);
        Wayland_init_xdg_output(d);
    } else if (SDL_strcmp(interface, "wp_viewporter") == 0) {
        d->viewporter = wl_registry_bind(d->registry, id, &wp_viewporter_interface, 1);
    } else if (SDL_strcmp(interface, "wp_fractional_scale_manager_v1") == 0) {
        d->fractional_scale_manager = wl_registry_bind(d->registry, id, &wp_fractional_scale_manager_v1_interface, 1);
    } else if (SDL_strcmp(interface, "zwp_input_timestamps_manager_v1") == 0) {
        d->input_timestamps_manager = wl_registry_bind(d->registry, id, &zwp_input_timestamps_manager_v1_interface, 1);
        Wayland_DisplayInitInputTimestampManager(d);
    } else if (SDL_strcmp(interface, "wp_cursor_shape_manager_v1") == 0) {
        d->cursor_shape_manager = wl_registry_bind(d->registry, id, &wp_cursor_shape_manager_v1_interface, 1);
        Wayland_DisplayInitCursorShapeManager(d);
    } else if (SDL_strcmp(interface, "zxdg_exporter_v2") == 0) {
        d->zxdg_exporter_v2 = wl_registry_bind(d->registry, id, &zxdg_exporter_v2_interface, 1);
    } else if (SDL_strcmp(interface, "xdg_wm_dialog_v1") == 0) {
        d->xdg_wm_dialog_v1 = wl_registry_bind(d->registry, id, &xdg_wm_dialog_v1_interface, 1);
    } else if (SDL_strcmp(interface, "wp_alpha_modifier_v1") == 0) {
        d->wp_alpha_modifier_v1 = wl_registry_bind(d->registry, id, &wp_alpha_modifier_v1_interface, 1);
    } else if (SDL_strcmp(interface, "xdg_toplevel_icon_manager_v1") == 0) {
        d->xdg_toplevel_icon_manager_v1 = wl_registry_bind(d->registry, id, &xdg_toplevel_icon_manager_v1_interface, 1);
    } else if (SDL_strcmp(interface, "frog_color_management_factory_v1") == 0) {
        d->frog_color_management_factory_v1 = wl_registry_bind(d->registry, id, &frog_color_management_factory_v1_interface, 1);
    } else if (SDL_strcmp(interface, "wp_color_manager_v1") == 0) {
        d->wp_color_manager_v1 = wl_registry_bind(d->registry, id, &wp_color_manager_v1_interface, 1);
        Wayland_InitColorManager(d);
    }
}

static void display_remove_global(void *data, struct wl_registry *registry, uint32_t id)
{
    SDL_VideoData *d = data;

    // We don't get an interface, just an ID, so check outputs and seats.
    for (int i = 0; i < d->output_count; ++i) {
        SDL_DisplayData *disp = d->output_list[i];
        if (disp->registry_id == id) {
            Wayland_free_display(SDL_GetVideoDisplay(disp->display), true);

            if (i < d->output_count) {
                SDL_memmove(&d->output_list[i], &d->output_list[i + 1], sizeof(SDL_DisplayData *) * (d->output_count - i - 1));
            }

            d->output_count--;
            return;
        }
    }

    struct SDL_WaylandSeat *seat, *temp;
    wl_list_for_each_safe (seat, temp, &d->seat_list, link)
    {
        if (seat->registry_id == id) {
            if (seat->keyboard.wl_keyboard) {
                SDL_RemoveKeyboard(seat->keyboard.sdl_id, true);
            }
            if (seat->keyboard.wl_keyboard) {
                SDL_RemoveMouse(seat->pointer.sdl_id, true);
            }
            Wayland_SeatDestroy(seat, true);
        }
    }
}

static const struct wl_registry_listener registry_listener = {
    display_handle_global,
    display_remove_global
};

#ifdef HAVE_LIBDECOR_H
static bool should_use_libdecor(SDL_VideoData *data, bool ignore_xdg)
{
    if (!SDL_WAYLAND_HAVE_WAYLAND_LIBDECOR) {
        return false;
    }

    if (!SDL_GetHintBoolean(SDL_HINT_VIDEO_WAYLAND_ALLOW_LIBDECOR, true)) {
        return false;
    }

    if (SDL_GetHintBoolean(SDL_HINT_VIDEO_WAYLAND_PREFER_LIBDECOR, false)) {
        return true;
    }

    if (ignore_xdg) {
        return true;
    }

    if (data->decoration_manager) {
        return false;
    }

    return true;
}
#endif

bool Wayland_LoadLibdecor(SDL_VideoData *data, bool ignore_xdg)
{
#ifdef HAVE_LIBDECOR_H
    if (data->shell.libdecor != NULL) {
        return true; // Already loaded!
    }
    if (should_use_libdecor(data, ignore_xdg)) {
        data->shell.libdecor = libdecor_new(data->display, &libdecor_interface);
        return data->shell.libdecor != NULL;
    }
#endif
    return false;
}

bool Wayland_VideoInit(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;

    data->xkb_context = WAYLAND_xkb_context_new(0);
    if (!data->xkb_context) {
        return SDL_SetError("Failed to create XKB context");
    }

    data->registry = wl_display_get_registry(data->display);
    if (!data->registry) {
        return SDL_SetError("Failed to get the Wayland registry");
    }

    wl_registry_add_listener(data->registry, &registry_listener, data);

    // First roundtrip to receive all registry objects.
    WAYLAND_wl_display_roundtrip(data->display);

    // Require viewports and xdg-output for display scaling.
    if (data->scale_to_display_enabled) {
        if (!data->viewporter) {
            SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "wayland: Display scaling requires the missing 'wp_viewporter' protocol: disabling");
            data->scale_to_display_enabled = false;
        }
        if (!data->xdg_output_manager) {
            SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "wayland: Display scaling requires the missing 'zxdg_output_manager_v1' protocol: disabling");
            data->scale_to_display_enabled = false;
        }
    }

    // Now that we have all the protocols, load libdecor if applicable
    Wayland_LoadLibdecor(data, false);

    // Second roundtrip to receive all output events.
    WAYLAND_wl_display_roundtrip(data->display);

    Wayland_FinalizeDisplays(data);

    Wayland_InitMouse();

    WAYLAND_wl_display_flush(data->display);

    Wayland_InitKeyboard(_this);

    if (data->primary_selection_device_manager) {
        _this->SetPrimarySelectionText = Wayland_SetPrimarySelectionText;
        _this->GetPrimarySelectionText = Wayland_GetPrimarySelectionText;
        _this->HasPrimarySelectionText = Wayland_HasPrimarySelectionText;
    }

    data->initializing = false;

    return true;
}

static bool Wayland_GetDisplayBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect)
{
    SDL_VideoData *viddata = _this->internal;
    SDL_DisplayData *internal = display->internal;
    rect->x = internal->x;
    rect->y = internal->y;

    // When an emulated, exclusive fullscreen window has focus, treat the mode dimensions as the display bounds.
    if (display->fullscreen_window &&
        display->fullscreen_window->fullscreen_exclusive &&
        display->fullscreen_window->internal->active &&
        display->fullscreen_window->current_fullscreen_mode.w != 0 &&
        display->fullscreen_window->current_fullscreen_mode.h != 0) {
        rect->w = display->fullscreen_window->current_fullscreen_mode.w;
        rect->h = display->fullscreen_window->current_fullscreen_mode.h;
    } else {
        if (!viddata->scale_to_display_enabled) {
            rect->w = display->current_mode->w;
            rect->h = display->current_mode->h;
        } else if (internal->transform & WL_OUTPUT_TRANSFORM_90) {
            rect->w = internal->pixel_height;
            rect->h = internal->pixel_width;
        } else {
            rect->w = internal->pixel_width;
            rect->h = internal->pixel_height;
        }
    }
    return true;
}

static void Wayland_VideoCleanup(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;
    SDL_WaylandSeat *seat, *tmp;
    int i;

    Wayland_FiniMouse(data);

    for (i = _this->num_displays - 1; i >= 0; --i) {
        SDL_VideoDisplay *display = _this->displays[i];
        Wayland_free_display(display, false);
    }
    SDL_free(data->output_list);

    wl_list_for_each_safe (seat, tmp, &data->seat_list, link) {
        Wayland_SeatDestroy(seat, false);
    }

    if (data->pointer_constraints) {
        zwp_pointer_constraints_v1_destroy(data->pointer_constraints);
        data->pointer_constraints = NULL;
    }

    if (data->relative_pointer_manager) {
        zwp_relative_pointer_manager_v1_destroy(data->relative_pointer_manager);
        data->relative_pointer_manager = NULL;
    }

    if (data->activation_manager) {
        xdg_activation_v1_destroy(data->activation_manager);
        data->activation_manager = NULL;
    }

    if (data->idle_inhibit_manager) {
        zwp_idle_inhibit_manager_v1_destroy(data->idle_inhibit_manager);
        data->idle_inhibit_manager = NULL;
    }

    if (data->key_inhibitor_manager) {
        zwp_keyboard_shortcuts_inhibit_manager_v1_destroy(data->key_inhibitor_manager);
        data->key_inhibitor_manager = NULL;
    }

    Wayland_QuitKeyboard(_this);

    if (data->text_input_manager) {
        zwp_text_input_manager_v3_destroy(data->text_input_manager);
        data->text_input_manager = NULL;
    }

    if (data->xkb_context) {
        WAYLAND_xkb_context_unref(data->xkb_context);
        data->xkb_context = NULL;
    }

    if (data->tablet_manager) {
        zwp_tablet_manager_v2_destroy((struct zwp_tablet_manager_v2 *)data->tablet_manager);
        data->tablet_manager = NULL;
    }

    if (data->data_device_manager) {
        wl_data_device_manager_destroy(data->data_device_manager);
        data->data_device_manager = NULL;
    }

    if (data->shm) {
        wl_shm_destroy(data->shm);
        data->shm = NULL;
    }

    if (data->shell.xdg) {
        xdg_wm_base_destroy(data->shell.xdg);
        data->shell.xdg = NULL;
    }

    if (data->decoration_manager) {
        zxdg_decoration_manager_v1_destroy(data->decoration_manager);
        data->decoration_manager = NULL;
    }

    if (data->xdg_output_manager) {
        zxdg_output_manager_v1_destroy(data->xdg_output_manager);
        data->xdg_output_manager = NULL;
    }

    if (data->viewporter) {
        wp_viewporter_destroy(data->viewporter);
        data->viewporter = NULL;
    }

    if (data->primary_selection_device_manager) {
        zwp_primary_selection_device_manager_v1_destroy(data->primary_selection_device_manager);
        data->primary_selection_device_manager = NULL;
    }

    if (data->fractional_scale_manager) {
        wp_fractional_scale_manager_v1_destroy(data->fractional_scale_manager);
        data->fractional_scale_manager = NULL;
    }

    if (data->input_timestamps_manager) {
        zwp_input_timestamps_manager_v1_destroy(data->input_timestamps_manager);
        data->input_timestamps_manager = NULL;
    }

    if (data->cursor_shape_manager) {
        wp_cursor_shape_manager_v1_destroy(data->cursor_shape_manager);
        data->cursor_shape_manager = NULL;
    }

    if (data->zxdg_exporter_v2) {
        zxdg_exporter_v2_destroy(data->zxdg_exporter_v2);
        data->zxdg_exporter_v2 = NULL;
    }

    if (data->xdg_wm_dialog_v1) {
        xdg_wm_dialog_v1_destroy(data->xdg_wm_dialog_v1);
        data->xdg_wm_dialog_v1 = NULL;
    }

    if (data->wp_alpha_modifier_v1) {
        wp_alpha_modifier_v1_destroy(data->wp_alpha_modifier_v1);
        data->wp_alpha_modifier_v1 = NULL;
    }

    if (data->xdg_toplevel_icon_manager_v1) {
        xdg_toplevel_icon_manager_v1_destroy(data->xdg_toplevel_icon_manager_v1);
        data->xdg_toplevel_icon_manager_v1 = NULL;
    }

    if (data->frog_color_management_factory_v1) {
        frog_color_management_factory_v1_destroy(data->frog_color_management_factory_v1);
        data->frog_color_management_factory_v1 = NULL;
    }

    if (data->wp_color_manager_v1) {
        wp_color_manager_v1_destroy(data->wp_color_manager_v1);
        data->wp_color_manager_v1 = NULL;
    }

    if (data->compositor) {
        wl_compositor_destroy(data->compositor);
        data->compositor = NULL;
    }

    if (data->registry) {
        wl_registry_destroy(data->registry);
        data->registry = NULL;
    }
}

bool Wayland_VideoReconnect(SDL_VideoDevice *_this)
{
#if 0 // TODO RECONNECT: Uncomment all when https://invent.kde.org/plasma/kwin/-/wikis/Restarting is completed
    SDL_VideoData *data = _this->internal;

    SDL_Window *window = NULL;

    SDL_GLContext current_ctx = SDL_GL_GetCurrentContext();
    SDL_Window *current_window = SDL_GL_GetCurrentWindow();

    SDL_GL_MakeCurrent(NULL, NULL);
    Wayland_VideoCleanup(_this);

    SDL_ResetKeyboard();
    SDL_ResetMouse();
    if (WAYLAND_wl_display_reconnect(data->display) < 0) {
        return false;
    }

    Wayland_VideoInit(_this);

    window = _this->windows;
    while (window) {
        /* We're going to cheat _just_ for a second and strip the OpenGL flag.
         * The Wayland driver actually forces it in CreateWindow, and
         * RecreateWindow does a bunch of unloading/loading of libGL, so just
         * strip the flag so RecreateWindow doesn't mess with the GL context,
         * and CreateWindow will add it right back!
         * -flibit
         */
        window->flags &= ~SDL_WINDOW_OPENGL;

        SDL_RecreateWindow(window, window->flags);
        window = window->next;
    }

    Wayland_RecreateCursors();

    if (current_window && current_ctx) {
        SDL_GL_MakeCurrent (current_window, current_ctx);
    }
    return true;
#else
    return false;
#endif // 0
}

void Wayland_VideoQuit(SDL_VideoDevice *_this)
{
    Wayland_VideoCleanup(_this);

#ifdef HAVE_LIBDECOR_H
    SDL_VideoData *data = _this->internal;
    if (data->shell.libdecor) {
        libdecor_unref(data->shell.libdecor);
        data->shell.libdecor = NULL;
    }
#endif
}

#endif // SDL_VIDEO_DRIVER_WAYLAND
