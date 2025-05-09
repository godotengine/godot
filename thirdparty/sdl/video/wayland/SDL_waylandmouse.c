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

#include "../SDL_sysvideo.h"
#include "../SDL_video_c.h"

#include "../../events/SDL_mouse_c.h"
#include "SDL_waylandvideo.h"
#include "../SDL_pixels_c.h"
#include "SDL_waylandevents_c.h"

#include "wayland-cursor.h"
#include "SDL_waylandmouse.h"
#include "SDL_waylandshmbuffer.h"

#include "cursor-shape-v1-client-protocol.h"
#include "pointer-constraints-unstable-v1-client-protocol.h"
#include "viewporter-client-protocol.h"

#include "../../SDL_hints_c.h"

static SDL_Cursor *sys_cursors[SDL_HITTEST_RESIZE_LEFT + 1];

static bool Wayland_SetRelativeMouseMode(bool enabled);

typedef struct
{
    struct Wayland_SHMBuffer shmBuffer;
    double scale;
    struct wl_list node;
} Wayland_ScaledCustomCursor;

typedef struct
{
    SDL_Surface *sdl_cursor_surface;
    int hot_x;
    int hot_y;
    struct wl_list scaled_cursor_cache;
} Wayland_CustomCursor;

typedef struct
{
    struct wl_buffer *wl_buffer;
    Uint64 duration_ns;
} Wayland_SystemCursorFrame;

typedef struct
{
    Wayland_SystemCursorFrame *frames;
    Uint64 total_duration_ns;
    int num_frames;
    SDL_SystemCursor id;
} Wayland_SystemCursor;

struct SDL_CursorData
{
    union
    {
        Wayland_CustomCursor custom;
        Wayland_SystemCursor system;
    } cursor_data;

    bool is_system_cursor;
};

static int dbus_cursor_size;
static char *dbus_cursor_theme;

static void Wayland_FreeCursorThemes(SDL_VideoData *vdata)
{
    for (int i = 0; i < vdata->num_cursor_themes; i += 1) {
        WAYLAND_wl_cursor_theme_destroy(vdata->cursor_themes[i].theme);
    }
    vdata->num_cursor_themes = 0;
    SDL_free(vdata->cursor_themes);
    vdata->cursor_themes = NULL;
}

#ifdef SDL_USE_LIBDBUS

#include "../../core/linux/SDL_dbus.h"

#define CURSOR_NODE        "org.freedesktop.portal.Desktop"
#define CURSOR_PATH        "/org/freedesktop/portal/desktop"
#define CURSOR_INTERFACE   "org.freedesktop.portal.Settings"
#define CURSOR_NAMESPACE   "org.gnome.desktop.interface"
#define CURSOR_SIGNAL_NAME "SettingChanged"
#define CURSOR_SIZE_KEY    "cursor-size"
#define CURSOR_THEME_KEY   "cursor-theme"

static DBusMessage *Wayland_ReadDBusProperty(SDL_DBusContext *dbus, const char *key)
{
    static const char *iface = "org.gnome.desktop.interface";

    DBusMessage *reply = NULL;
    DBusMessage *msg = dbus->message_new_method_call(CURSOR_NODE,
                                                     CURSOR_PATH,
                                                     CURSOR_INTERFACE,
                                                     "Read"); // Method

    if (msg) {
        if (dbus->message_append_args(msg, DBUS_TYPE_STRING, &iface, DBUS_TYPE_STRING, &key, DBUS_TYPE_INVALID)) {
            reply = dbus->connection_send_with_reply_and_block(dbus->session_conn, msg, DBUS_TIMEOUT_USE_DEFAULT, NULL);
        }
        dbus->message_unref(msg);
    }

    return reply;
}

static bool Wayland_ParseDBusReply(SDL_DBusContext *dbus, DBusMessage *reply, int type, void *value)
{
    DBusMessageIter iter[3];

    dbus->message_iter_init(reply, &iter[0]);
    if (dbus->message_iter_get_arg_type(&iter[0]) != DBUS_TYPE_VARIANT) {
        return false;
    }

    dbus->message_iter_recurse(&iter[0], &iter[1]);
    if (dbus->message_iter_get_arg_type(&iter[1]) != DBUS_TYPE_VARIANT) {
        return false;
    }

    dbus->message_iter_recurse(&iter[1], &iter[2]);
    if (dbus->message_iter_get_arg_type(&iter[2]) != type) {
        return false;
    }

    dbus->message_iter_get_basic(&iter[2], value);

    return true;
}

static DBusHandlerResult Wayland_DBusCursorMessageFilter(DBusConnection *conn, DBusMessage *msg, void *data)
{
    SDL_DBusContext *dbus = SDL_DBus_GetContext();
    SDL_VideoData *vdata = (SDL_VideoData *)data;

    if (dbus->message_is_signal(msg, CURSOR_INTERFACE, CURSOR_SIGNAL_NAME)) {
        DBusMessageIter signal_iter, variant_iter;
        const char *namespace, *key;

        dbus->message_iter_init(msg, &signal_iter);
        // Check if the parameters are what we expect
        if (dbus->message_iter_get_arg_type(&signal_iter) != DBUS_TYPE_STRING) {
            goto not_our_signal;
        }
        dbus->message_iter_get_basic(&signal_iter, &namespace);
        if (SDL_strcmp(CURSOR_NAMESPACE, namespace) != 0) {
            goto not_our_signal;
        }
        if (!dbus->message_iter_next(&signal_iter)) {
            goto not_our_signal;
        }
        if (dbus->message_iter_get_arg_type(&signal_iter) != DBUS_TYPE_STRING) {
            goto not_our_signal;
        }
        dbus->message_iter_get_basic(&signal_iter, &key);
        if (SDL_strcmp(CURSOR_SIZE_KEY, key) == 0) {
            int new_cursor_size;

            if (!dbus->message_iter_next(&signal_iter)) {
                goto not_our_signal;
            }
            if (dbus->message_iter_get_arg_type(&signal_iter) != DBUS_TYPE_VARIANT) {
                goto not_our_signal;
            }
            dbus->message_iter_recurse(&signal_iter, &variant_iter);
            if (dbus->message_iter_get_arg_type(&variant_iter) != DBUS_TYPE_INT32) {
                goto not_our_signal;
            }
            dbus->message_iter_get_basic(&variant_iter, &new_cursor_size);

            if (dbus_cursor_size != new_cursor_size) {
                dbus_cursor_size = new_cursor_size;
                SDL_RedrawCursor(); // Force cursor update
            }
        } else if (SDL_strcmp(CURSOR_THEME_KEY, key) == 0) {
            const char *new_cursor_theme = NULL;

            if (!dbus->message_iter_next(&signal_iter)) {
                goto not_our_signal;
            }
            if (dbus->message_iter_get_arg_type(&signal_iter) != DBUS_TYPE_VARIANT) {
                goto not_our_signal;
            }
            dbus->message_iter_recurse(&signal_iter, &variant_iter);
            if (dbus->message_iter_get_arg_type(&variant_iter) != DBUS_TYPE_STRING) {
                goto not_our_signal;
            }
            dbus->message_iter_get_basic(&variant_iter, &new_cursor_theme);

            if (!dbus_cursor_theme || !new_cursor_theme || SDL_strcmp(dbus_cursor_theme, new_cursor_theme) != 0) {
                SDL_free(dbus_cursor_theme);
                if (new_cursor_theme) {
                    dbus_cursor_theme = SDL_strdup(new_cursor_theme);
                } else {
                    dbus_cursor_theme = NULL;
                }

                // Purge the current cached themes and force a cursor refresh.
                Wayland_FreeCursorThemes(vdata);
                SDL_RedrawCursor();
            }
        } else {
            goto not_our_signal;
        }

        return DBUS_HANDLER_RESULT_HANDLED;
    }

not_our_signal:
    return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
}

static void Wayland_DBusInitCursorProperties(SDL_VideoData *vdata)
{
    DBusMessage *reply;
    SDL_DBusContext *dbus = SDL_DBus_GetContext();
    bool add_filter = false;

    if (!dbus) {
        return;
    }

    if ((reply = Wayland_ReadDBusProperty(dbus, CURSOR_SIZE_KEY))) {
        if (Wayland_ParseDBusReply(dbus, reply, DBUS_TYPE_INT32, &dbus_cursor_size)) {
            add_filter = true;
        }
        dbus->message_unref(reply);
    }

    if ((reply = Wayland_ReadDBusProperty(dbus, CURSOR_THEME_KEY))) {
        const char *temp = NULL;
        if (Wayland_ParseDBusReply(dbus, reply, DBUS_TYPE_STRING, &temp)) {
            add_filter = true;

            if (temp) {
                dbus_cursor_theme = SDL_strdup(temp);
            }
        }
        dbus->message_unref(reply);
    }

    // Only add the filter if at least one of the settings we want is present.
    if (add_filter) {
        dbus->bus_add_match(dbus->session_conn,
                            "type='signal', interface='" CURSOR_INTERFACE "',"
                            "member='" CURSOR_SIGNAL_NAME "', arg0='" CURSOR_NAMESPACE "'",
                            NULL);
        dbus->connection_add_filter(dbus->session_conn, &Wayland_DBusCursorMessageFilter, vdata, NULL);
        dbus->connection_flush(dbus->session_conn);
    }
}

static void Wayland_DBusFinishCursorProperties(void)
{
    SDL_free(dbus_cursor_theme);
    dbus_cursor_theme = NULL;
}

#endif

static void cursor_frame_done(void *data, struct wl_callback *cb, uint32_t time);
struct wl_callback_listener cursor_frame_listener = {
    cursor_frame_done
};

static void cursor_frame_done(void *data, struct wl_callback *cb, uint32_t time)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    SDL_CursorData *c = (struct SDL_CursorData *)seat->pointer.current_cursor;

    const Uint64 now = SDL_GetTicksNS();
    const Uint64 elapsed = (now - seat->pointer.cursor_state.last_frame_callback_time_ns) % c->cursor_data.system.total_duration_ns;
    Uint64 advance = 0;
    int next = seat->pointer.cursor_state.current_frame;

    wl_callback_destroy(cb);
    seat->pointer.cursor_state.frame_callback = wl_surface_frame(seat->pointer.cursor_state.surface);
    wl_callback_add_listener(seat->pointer.cursor_state.frame_callback, &cursor_frame_listener, data);

    seat->pointer.cursor_state.current_frame_time_ns += elapsed;

    // Calculate the next frame based on the elapsed duration.
    for (Uint64 t = c->cursor_data.system.frames[next].duration_ns; t <= seat->pointer.cursor_state.current_frame_time_ns; t += c->cursor_data.system.frames[next].duration_ns) {
        next = (next + 1) % c->cursor_data.system.num_frames;
        advance = t;

        // Make sure we don't end up in an infinite loop if a cursor has frame durations of 0.
        if (!c->cursor_data.system.frames[next].duration_ns) {
            break;
        }
    }

    seat->pointer.cursor_state.current_frame_time_ns -= advance;
    seat->pointer.cursor_state.last_frame_callback_time_ns = now;
    seat->pointer.cursor_state.current_frame = next;
    wl_surface_attach(seat->pointer.cursor_state.surface, c->cursor_data.system.frames[next].wl_buffer, 0, 0);
    if (wl_surface_get_version(seat->pointer.cursor_state.surface) >= WL_SURFACE_DAMAGE_BUFFER_SINCE_VERSION) {
        wl_surface_damage_buffer(seat->pointer.cursor_state.surface, 0, 0, SDL_MAX_SINT32, SDL_MAX_SINT32);
    } else {
        wl_surface_damage(seat->pointer.cursor_state.surface, 0, 0, SDL_MAX_SINT32, SDL_MAX_SINT32);
    }
    wl_surface_commit(seat->pointer.cursor_state.surface);
}

static bool Wayland_GetSystemCursor(SDL_VideoData *vdata, SDL_CursorData *cdata, int *scale, int *dst_size, int *hot_x, int *hot_y)
{
    struct wl_cursor_theme *theme = NULL;
    struct wl_cursor *cursor;
    const char *css_name = "default";
    const char *fallback_name = NULL;
    double scale_factor = 1.0;
    int theme_size = dbus_cursor_size;

    // Fallback envvar if the DBus properties don't exist
    if (theme_size <= 0) {
        const char *xcursor_size = SDL_getenv("XCURSOR_SIZE");
        if (xcursor_size) {
            theme_size = SDL_atoi(xcursor_size);
        }
    }
    if (theme_size <= 0) {
        theme_size = 24;
    }
    // First, find the appropriate theme based on the current scale...
    SDL_Window *focus = SDL_GetMouse()->focus;
    if (focus) {
        // TODO: Use the fractional scale once GNOME supports viewports on cursor surfaces.
        scale_factor = SDL_ceil(focus->internal->scale_factor);
    }

    const int scaled_size = (int)SDL_lround(theme_size * scale_factor);
    for (int i = 0; i < vdata->num_cursor_themes; ++i) {
        if (vdata->cursor_themes[i].size == scaled_size) {
            theme = vdata->cursor_themes[i].theme;
            break;
        }
    }
    if (!theme) {
        const char *xcursor_theme = dbus_cursor_theme;

        SDL_WaylandCursorTheme *new_cursor_themes = SDL_realloc(vdata->cursor_themes,
                                                                sizeof(SDL_WaylandCursorTheme) * (vdata->num_cursor_themes + 1));
        if (!new_cursor_themes) {
            return false;
        }
        vdata->cursor_themes = new_cursor_themes;

        // Fallback envvar if the DBus properties don't exist
        if (!xcursor_theme) {
            xcursor_theme = SDL_getenv("XCURSOR_THEME");
        }

        theme = WAYLAND_wl_cursor_theme_load(xcursor_theme, scaled_size, vdata->shm);
        vdata->cursor_themes[vdata->num_cursor_themes].size = scaled_size;
        vdata->cursor_themes[vdata->num_cursor_themes++].theme = theme;
    }

    css_name = SDL_GetCSSCursorName(cdata->cursor_data.system.id, &fallback_name);
    cursor = WAYLAND_wl_cursor_theme_get_cursor(theme, css_name);
    if (!cursor && fallback_name) {
        cursor = WAYLAND_wl_cursor_theme_get_cursor(theme, fallback_name);
    }

    // Fallback to the default cursor if the chosen one wasn't found
    if (!cursor) {
        cursor = WAYLAND_wl_cursor_theme_get_cursor(theme, "default");
    }
    // Try the old X11 name as a last resort
    if (!cursor) {
        cursor = WAYLAND_wl_cursor_theme_get_cursor(theme, "left_ptr");
    }
    if (!cursor) {
        return false;
    }

    if (cdata->cursor_data.system.num_frames != cursor->image_count) {
        SDL_free(cdata->cursor_data.system.frames);
        cdata->cursor_data.system.frames = SDL_calloc(cursor->image_count, sizeof(Wayland_SystemCursorFrame));
        if (!cdata->cursor_data.system.frames) {
            return false;
        }
    }

    // ... Set the cursor data, finally.
    cdata->cursor_data.system.num_frames = cursor->image_count;
    cdata->cursor_data.system.total_duration_ns = 0;
    for (int i = 0; i < cursor->image_count; ++i) {
        cdata->cursor_data.system.frames[i].wl_buffer = WAYLAND_wl_cursor_image_get_buffer(cursor->images[i]);
        cdata->cursor_data.system.frames[i].duration_ns = SDL_MS_TO_NS((Uint64)cursor->images[i]->delay);
        cdata->cursor_data.system.total_duration_ns += cdata->cursor_data.system.frames[i].duration_ns;
    }

    *scale = SDL_ceil(scale_factor) == scale_factor ? (int)scale_factor : 0;

    if (scaled_size != cursor->images[0]->width) {
        /* If the cursor size isn't an exact match for the target size, use a viewport
         * to avoid a possible "Buffer size is not divisible by scale" protocol error.
         *
         * If viewports are unavailable, find an integer scale that works.
         */
        if (vdata->viewporter) {
            // A scale of 0 indicates that a viewport set to the destination size should be used.
            *scale = 0;
        } else {
            for (; *scale > 1; --*scale) {
                if (cursor->images[0]->width % *scale == 0) {
                    break;
                }
            }
            // Set the scale factor to the new value for the hotspot calculations.
            scale_factor = *scale;
        }
    }

    *dst_size = (int)SDL_lround(cursor->images[0]->width / scale_factor);

    *hot_x = (int)SDL_lround(cursor->images[0]->hotspot_x / scale_factor);
    *hot_y = (int)SDL_lround(cursor->images[0]->hotspot_y / scale_factor);

    return true;
}

static Wayland_ScaledCustomCursor *Wayland_CacheScaledCustomCursor(SDL_CursorData *cdata, double scale)
{
    Wayland_ScaledCustomCursor *cache = NULL;

    // Is this cursor already cached at the target scale?
    if (!WAYLAND_wl_list_empty(&cdata->cursor_data.custom.scaled_cursor_cache)) {
        Wayland_ScaledCustomCursor *c = NULL;
        wl_list_for_each (c, &cdata->cursor_data.custom.scaled_cursor_cache, node) {
            if (c->scale == scale) {
                cache = c;
                break;
            }
        }
    }

    if (!cache) {
        cache = SDL_calloc(1, sizeof(Wayland_ScaledCustomCursor));
        if (!cache) {
            return NULL;
        }

        SDL_Surface *surface = SDL_GetSurfaceImage(cdata->cursor_data.custom.sdl_cursor_surface, (float)scale);
        if (!surface) {
            SDL_free(cache);
            return NULL;
        }

        // Allocate the shared memory buffer for this cursor.
        if (!Wayland_AllocSHMBuffer(surface->w, surface->h, &cache->shmBuffer)) {
            SDL_free(cache);
            SDL_DestroySurface(surface);
            return NULL;
        }

        // Wayland requires premultiplied alpha for its surfaces.
        SDL_PremultiplyAlpha(surface->w, surface->h,
                             surface->format, surface->pixels, surface->pitch,
                             SDL_PIXELFORMAT_ARGB8888, cache->shmBuffer.shm_data, surface->w * 4, true);

        cache->scale = scale;
        WAYLAND_wl_list_insert(&cdata->cursor_data.custom.scaled_cursor_cache, &cache->node);
        SDL_DestroySurface(surface);
    }

    return cache;
}

static bool Wayland_GetCustomCursor(SDL_Cursor *cursor, struct wl_buffer **buffer, int *scale, int *dst_width, int *dst_height, int *hot_x, int *hot_y)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *wd = vd->internal;
    SDL_CursorData *data = cursor->internal;
    SDL_Window *focus = SDL_GetMouseFocus();
    double scale_factor = 1.0;

    if (focus && SDL_SurfaceHasAlternateImages(data->cursor_data.custom.sdl_cursor_surface)) {
        scale_factor = focus->internal->scale_factor;
    }

    // Only use fractional scale values if viewports are available.
    if (!wd->viewporter) {
        scale_factor = SDL_ceil(scale_factor);
    }

    Wayland_ScaledCustomCursor *c = Wayland_CacheScaledCustomCursor(data, scale_factor);
    if (!c) {
        return false;
    }

    *buffer = c->shmBuffer.wl_buffer;
    *scale = SDL_ceil(scale_factor) == scale_factor ? (int)scale_factor : 0;
    *dst_width = data->cursor_data.custom.sdl_cursor_surface->w;
    *dst_height = data->cursor_data.custom.sdl_cursor_surface->h;
    *hot_x = data->cursor_data.custom.hot_x;
    *hot_y = data->cursor_data.custom.hot_y;

    return true;
}

static SDL_Cursor *Wayland_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    SDL_Cursor *cursor = SDL_calloc(1, sizeof(*cursor));

    if (cursor) {
        SDL_CursorData *data = SDL_calloc(1, sizeof(*data));
        if (!data) {
            SDL_free(cursor);
            return NULL;
        }
        cursor->internal = data;
        WAYLAND_wl_list_init(&data->cursor_data.custom.scaled_cursor_cache);
        data->cursor_data.custom.hot_x = hot_x;
        data->cursor_data.custom.hot_y = hot_y;

        data->cursor_data.custom.sdl_cursor_surface = surface;
        ++surface->refcount;

        // If the cursor has only one size, just prepare it now.
        if (!SDL_SurfaceHasAlternateImages(surface)) {
            Wayland_CacheScaledCustomCursor(data, 1.0);
        }
    }

    return cursor;
}

static SDL_Cursor *Wayland_CreateSystemCursor(SDL_SystemCursor id)
{
    SDL_Cursor *cursor = SDL_calloc(1, sizeof(*cursor));

    if (cursor) {
        SDL_CursorData *cdata = SDL_calloc(1, sizeof(*cdata));
        if (!cdata) {
            SDL_free(cursor);
            return NULL;
        }
        cursor->internal = cdata;

        cdata->cursor_data.system.id = id;
        cdata->is_system_cursor = true;
    }

    return cursor;
}

static SDL_Cursor *Wayland_CreateDefaultCursor(void)
{
    SDL_SystemCursor id = SDL_GetDefaultSystemCursor();
    return Wayland_CreateSystemCursor(id);
}

static void Wayland_FreeCursorData(SDL_CursorData *d)
{
    SDL_VideoDevice *video_device = SDL_GetVideoDevice();
    SDL_VideoData *video_data = video_device->internal;
    SDL_WaylandSeat *seat;

    // Stop any frame callbacks and detach buffers associated with the cursor being destroyed.
    wl_list_for_each (seat, &video_data->seat_list, link)
    {
        if (seat->pointer.current_cursor == d) {
            if (seat->pointer.cursor_state.frame_callback) {
                wl_callback_destroy(seat->pointer.cursor_state.frame_callback);
                seat->pointer.cursor_state.frame_callback = NULL;
            }
            if (seat->pointer.cursor_state.surface) {
                wl_surface_attach(seat->pointer.cursor_state.surface, NULL, 0, 0);
            }

            seat->pointer.current_cursor = NULL;
        }
    }

    // Buffers for system cursors must not be destroyed.
    if (d->is_system_cursor) {
        SDL_free(d->cursor_data.system.frames);
    } else {
        Wayland_ScaledCustomCursor *c, *temp;
        wl_list_for_each_safe(c, temp, &d->cursor_data.custom.scaled_cursor_cache, node) {
            Wayland_ReleaseSHMBuffer(&c->shmBuffer);
            SDL_free(c);
        }

        SDL_DestroySurface(d->cursor_data.custom.sdl_cursor_surface);
    }
}

static void Wayland_FreeCursor(SDL_Cursor *cursor)
{
    if (!cursor) {
        return;
    }

    // Probably not a cursor we own
    if (!cursor->internal) {
        return;
    }

    Wayland_FreeCursorData(cursor->internal);

    SDL_free(cursor->internal);
    SDL_free(cursor);
}

static void Wayland_SetSystemCursorShape(SDL_WaylandSeat *seat, SDL_SystemCursor id)
{
    Uint32 shape;

    switch (id) {
    case SDL_SYSTEM_CURSOR_DEFAULT:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_DEFAULT;
        break;
    case SDL_SYSTEM_CURSOR_TEXT:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_TEXT;
        break;
    case SDL_SYSTEM_CURSOR_WAIT:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_WAIT;
        break;
    case SDL_SYSTEM_CURSOR_CROSSHAIR:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_CROSSHAIR;
        break;
    case SDL_SYSTEM_CURSOR_PROGRESS:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_PROGRESS;
        break;
    case SDL_SYSTEM_CURSOR_NWSE_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_NWSE_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_NESW_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_NESW_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_EW_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_EW_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_NS_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_NS_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_MOVE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_ALL_SCROLL;
        break;
    case SDL_SYSTEM_CURSOR_NOT_ALLOWED:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_NOT_ALLOWED;
        break;
    case SDL_SYSTEM_CURSOR_POINTER:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_POINTER;
        break;
    case SDL_SYSTEM_CURSOR_NW_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_NW_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_N_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_N_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_NE_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_NE_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_E_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_E_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_SE_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_SE_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_S_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_S_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_SW_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_SW_RESIZE;
        break;
    case SDL_SYSTEM_CURSOR_W_RESIZE:
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_W_RESIZE;
        break;
    default:
        SDL_assert(0); // Should never be here...
        shape = WP_CURSOR_SHAPE_DEVICE_V1_SHAPE_DEFAULT;
    }

    wp_cursor_shape_device_v1_set_shape(seat->pointer.cursor_shape, seat->pointer.enter_serial, shape);
}

static void Wayland_SeatSetCursor(SDL_WaylandSeat *seat, SDL_Cursor *cursor)
{
    if (seat->pointer.wl_pointer) {
        struct wl_buffer *buffer = NULL;
        int scale = 1;
        int dst_width = 0;
        int dst_height = 0;
        int hot_x;
        int hot_y;
        SDL_CursorData *cursor_data = cursor ? cursor->internal : NULL;

        // Stop the frame callback for old animated cursors.
        if (seat->pointer.cursor_state.frame_callback && cursor_data != seat->pointer.current_cursor) {
            wl_callback_destroy(seat->pointer.cursor_state.frame_callback);
            seat->pointer.cursor_state.frame_callback = NULL;
        }

        if (cursor) {
            if (cursor_data == seat->pointer.current_cursor) {
                return;
            }

            if (cursor_data->is_system_cursor) {
                // If the cursor shape protocol is supported, the compositor will draw nicely scaled cursors for us, so nothing more to do.
                if (seat->pointer.cursor_shape) {
                    // Don't need the surface or viewport if using the cursor shape protocol.
                    if (seat->pointer.cursor_state.surface) {
                        wl_pointer_set_cursor(seat->pointer.wl_pointer, seat->pointer.enter_serial, NULL, 0, 0);
                        wl_surface_destroy(seat->pointer.cursor_state.surface);
                        seat->pointer.cursor_state.surface = NULL;
                    }
                    if (seat->pointer.cursor_state.viewport) {
                        wp_viewport_destroy(seat->pointer.cursor_state.viewport);
                        seat->pointer.cursor_state.viewport = NULL;
                    }

                    Wayland_SetSystemCursorShape(seat, cursor_data->cursor_data.system.id);
                    seat->pointer.current_cursor = cursor_data;

                    return;
                }

                if (!Wayland_GetSystemCursor(seat->display, cursor_data, &scale, &dst_width, &hot_x, &hot_y)) {
                    return;
                }

                dst_height = dst_width;

                if (!seat->pointer.cursor_state.surface) {
                    seat->pointer.cursor_state.surface = wl_compositor_create_surface(seat->display->compositor);
                }
                wl_surface_attach(seat->pointer.cursor_state.surface, cursor_data->cursor_data.system.frames[0].wl_buffer, 0, 0);

                // If more than one frame is available, create a frame callback to run the animation.
                if (cursor_data->cursor_data.system.num_frames > 1) {
                    seat->pointer.cursor_state.last_frame_callback_time_ns = SDL_GetTicks();
                    seat->pointer.cursor_state.current_frame_time_ns = 0;
                    seat->pointer.cursor_state.current_frame = 0;
                    seat->pointer.cursor_state.frame_callback = wl_surface_frame(seat->pointer.cursor_state.surface);
                    wl_callback_add_listener(seat->pointer.cursor_state.frame_callback, &cursor_frame_listener, seat);
                }
            } else {
                if (!Wayland_GetCustomCursor(cursor, &buffer, &scale, &dst_width, &dst_height, &hot_x, &hot_y)) {
                    return;
                }

                if (!seat->pointer.cursor_state.surface) {
                    seat->pointer.cursor_state.surface = wl_compositor_create_surface(seat->display->compositor);
                }
                wl_surface_attach(seat->pointer.cursor_state.surface, buffer, 0, 0);
            }

            // A scale value of 0 indicates that a viewport with the returned destination size should be used.
            if (!scale) {
                if (!seat->pointer.cursor_state.viewport) {
                    seat->pointer.cursor_state.viewport = wp_viewporter_get_viewport(seat->display->viewporter, seat->pointer.cursor_state.surface);
                }
                wl_surface_set_buffer_scale(seat->pointer.cursor_state.surface, 1);
                wp_viewport_set_source(seat->pointer.cursor_state.viewport, wl_fixed_from_int(-1), wl_fixed_from_int(-1), wl_fixed_from_int(-1), wl_fixed_from_int(-1));
                wp_viewport_set_destination(seat->pointer.cursor_state.viewport, dst_width, dst_height);
            } else {
                if (seat->pointer.cursor_state.viewport) {
                    wp_viewport_destroy(seat->pointer.cursor_state.viewport);
                    seat->pointer.cursor_state.viewport = NULL;
                }
                wl_surface_set_buffer_scale(seat->pointer.cursor_state.surface, scale);
            }

            wl_pointer_set_cursor(seat->pointer.wl_pointer, seat->pointer.enter_serial, seat->pointer.cursor_state.surface, hot_x, hot_y);

            if (wl_surface_get_version(seat->pointer.cursor_state.surface) >= WL_SURFACE_DAMAGE_BUFFER_SINCE_VERSION) {
                wl_surface_damage_buffer(seat->pointer.cursor_state.surface, 0, 0, SDL_MAX_SINT32, SDL_MAX_SINT32);
            } else {
                wl_surface_damage(seat->pointer.cursor_state.surface, 0, 0, SDL_MAX_SINT32, SDL_MAX_SINT32);
            }

            seat->pointer.current_cursor = cursor_data;
            wl_surface_commit(seat->pointer.cursor_state.surface);
        } else {
            seat->pointer.current_cursor = NULL;
            wl_pointer_set_cursor(seat->pointer.wl_pointer, seat->pointer.enter_serial, NULL, 0, 0);
        }
    }
}

static bool Wayland_ShowCursor(SDL_Cursor *cursor)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *d = vd->internal;
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_WaylandSeat *seat;

    wl_list_for_each (seat, &d->seat_list, link) {
        if (mouse->focus && mouse->focus->internal == seat->pointer.focus) {
            Wayland_SeatSetCursor(seat, cursor);
        } else if (!seat->pointer.focus) {
            Wayland_SeatSetCursor(seat, NULL);
        }
    }

    return true;
}

void Wayland_SeatWarpMouse(SDL_WaylandSeat *seat, SDL_WindowData *window, float x, float y)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *d = vd->internal;

    if (seat->pointer.wl_pointer) {
        bool toggle_lock = !seat->pointer.locked_pointer;
        bool update_grabs = false;

        /* The pointer confinement protocol allows setting a hint to warp the pointer,
         * but only when the pointer is locked.
         *
         * Lock the pointer, set the position hint, unlock, and hope for the best.
         */
        if (toggle_lock) {
            if (seat->pointer.confined_pointer) {
                zwp_confined_pointer_v1_destroy(seat->pointer.confined_pointer);
                seat->pointer.confined_pointer = NULL;
                update_grabs = true;
            }
            seat->pointer.locked_pointer = zwp_pointer_constraints_v1_lock_pointer(d->pointer_constraints, window->surface,
                                                                                   seat->pointer.wl_pointer, NULL,
                                                                                   ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_ONESHOT);
        }

        const wl_fixed_t f_x = wl_fixed_from_double(x / window->pointer_scale.x);
        const wl_fixed_t f_y = wl_fixed_from_double(y / window->pointer_scale.y);
        zwp_locked_pointer_v1_set_cursor_position_hint(seat->pointer.locked_pointer, f_x, f_y);
        wl_surface_commit(window->surface);

        if (toggle_lock) {
            zwp_locked_pointer_v1_destroy(seat->pointer.locked_pointer);
            seat->pointer.locked_pointer = NULL;

            if (update_grabs) {
                Wayland_SeatUpdatePointerGrab(seat);
            }
        }

        /* NOTE: There is a pending warp event under discussion that should replace this when available.
         * https://gitlab.freedesktop.org/wayland/wayland/-/merge_requests/340
         */
        SDL_SendMouseMotion(0, window->sdlwindow, seat->pointer.sdl_id, false, x, y);
    }
}

static bool Wayland_WarpMouseRelative(SDL_Window *window, float x, float y)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *d = vd->internal;
    SDL_WindowData *wind = window->internal;
    SDL_WaylandSeat *seat;

    if (d->pointer_constraints) {
        wl_list_for_each (seat, &d->seat_list, link) {
            if (wind == seat->pointer.focus ||
                (!seat->pointer.focus && wind == seat->keyboard.focus)) {
                Wayland_SeatWarpMouse(seat, wind, x, y);
            }
        }
    } else {
        return SDL_SetError("wayland: mouse warp failed; compositor lacks support for the required zwp_pointer_confinement_v1 protocol");
    }

    return true;
}

static bool Wayland_WarpMouseGlobal(float x, float y)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *d = vd->internal;
    SDL_WaylandSeat *seat;

    if (d->pointer_constraints) {
        wl_list_for_each (seat, &d->seat_list, link) {
            SDL_WindowData *wind = seat->pointer.focus ? seat->pointer.focus : seat->keyboard.focus;

            // If the client wants the coordinates warped to within a focused window, just convert the coordinates to relative.
            if (wind) {
                SDL_Window *window = wind->sdlwindow;

                int abs_x, abs_y;
                SDL_RelativeToGlobalForWindow(window, window->x, window->y, &abs_x, &abs_y);

                const SDL_FPoint p = { x, y };
                const SDL_FRect r = { abs_x, abs_y, window->w, window->h };

                // Try to warp the cursor if the point is within the seat's focused window.
                if (SDL_PointInRectFloat(&p, &r)) {
                    Wayland_SeatWarpMouse(seat, wind, p.x - abs_x, p.y - abs_y);
                }
            }
        }
    } else {
        return SDL_SetError("wayland: mouse warp failed; compositor lacks support for the required zwp_pointer_confinement_v1 protocol");
    }

    return true;
}

static bool Wayland_SetRelativeMouseMode(bool enabled)
{
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *data = vd->internal;

    // Relative mode requires both the relative motion and pointer confinement protocols.
    if (!data->relative_pointer_manager) {
        return SDL_SetError("Failed to enable relative mode: compositor lacks support for the required zwp_relative_pointer_manager_v1 protocol");
    }
    if (!data->pointer_constraints) {
        return SDL_SetError("Failed to enable relative mode: compositor lacks support for the required zwp_pointer_constraints_v1 protocol");
    }

    data->relative_mode_enabled = enabled;
    Wayland_DisplayUpdatePointerGrabs(data, NULL);
    return true;
}

/* Wayland doesn't support getting the true global cursor position, but it can
 * be faked well enough for what most applications use it for: querying the
 * global cursor coordinates and transforming them to the window-relative
 * coordinates manually.
 *
 * The global position is derived by taking the cursor position relative to the
 * toplevel window, and offsetting it by the origin of the output the window is
 * currently considered to be on. The cursor position and button state when the
 * cursor is outside an application window are unknown, but this gives 'correct'
 * coordinates when the window has focus, which is good enough for most
 * applications.
 */
static SDL_MouseButtonFlags SDLCALL Wayland_GetGlobalMouseState(float *x, float *y)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_MouseButtonFlags result = 0;

    // If there is no window with mouse focus, we have no idea what the actual position or button state is.
    if (mouse->focus) {
        int off_x, off_y;
        SDL_RelativeToGlobalForWindow(mouse->focus, mouse->focus->x, mouse->focus->y, &off_x, &off_y);
        result = SDL_GetMouseState(x, y);
        *x = mouse->x + off_x;
        *y = mouse->y + off_y;
    } else {
        *x = 0.f;
        *y = 0.f;
    }

    return result;
}

#if 0  // TODO RECONNECT: See waylandvideo.c for more information!
static void Wayland_RecreateCursor(SDL_Cursor *cursor, SDL_VideoData *vdata)
{
    SDL_CursorData *cdata = cursor->internal;

    // Probably not a cursor we own
    if (cdata == NULL) {
        return;
    }

    Wayland_FreeCursorData(cdata);

    // We're not currently freeing this, so... yolo?
    if (cdata->shm_data != NULL) {
        void *old_data_pointer = cdata->shm_data;
        int stride = cdata->w * 4;

        create_buffer_from_shm(cdata, cdata->w, cdata->h, WL_SHM_FORMAT_ARGB8888);

        SDL_memcpy(cdata->shm_data, old_data_pointer, stride * cdata->h);
    }
    cdata->surface = wl_compositor_create_surface(vdata->compositor);
    wl_surface_set_user_data(cdata->surface, NULL);
}

void Wayland_RecreateCursors(void)
{
    SDL_Cursor *cursor;
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_VideoData *vdata = SDL_GetVideoDevice()->internal;

    if (vdata && vdata->cursor_themes) {
        SDL_free(vdata->cursor_themes);
        vdata->cursor_themes = NULL;
        vdata->num_cursor_themes = 0;
    }

    if (mouse == NULL) {
        return;
    }

    for (cursor = mouse->cursors; cursor != NULL; cursor = cursor->next) {
        Wayland_RecreateCursor(cursor, vdata);
    }
    if (mouse->def_cursor) {
        Wayland_RecreateCursor(mouse->def_cursor, vdata);
    }
    if (mouse->cur_cursor) {
        Wayland_RecreateCursor(mouse->cur_cursor, vdata);
        if (mouse->cursor_visible) {
            Wayland_ShowCursor(mouse->cur_cursor);
        }
    }
}
#endif // 0

void Wayland_InitMouse(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_VideoDevice *vd = SDL_GetVideoDevice();
    SDL_VideoData *d = vd->internal;

    mouse->CreateCursor = Wayland_CreateCursor;
    mouse->CreateSystemCursor = Wayland_CreateSystemCursor;
    mouse->ShowCursor = Wayland_ShowCursor;
    mouse->FreeCursor = Wayland_FreeCursor;
    mouse->WarpMouse = Wayland_WarpMouseRelative;
    mouse->WarpMouseGlobal = Wayland_WarpMouseGlobal;
    mouse->SetRelativeMouseMode = Wayland_SetRelativeMouseMode;
    mouse->GetGlobalMouseState = Wayland_GetGlobalMouseState;

    SDL_HitTestResult r = SDL_HITTEST_NORMAL;
    while (r <= SDL_HITTEST_RESIZE_LEFT) {
        switch (r) {
        case SDL_HITTEST_NORMAL:
            sys_cursors[r] = Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_DEFAULT);
            break;
        case SDL_HITTEST_DRAGGABLE:
            sys_cursors[r] = Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_DEFAULT);
            break;
        case SDL_HITTEST_RESIZE_TOPLEFT:
            sys_cursors[r] = Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_NW_RESIZE);
            break;
        case SDL_HITTEST_RESIZE_TOP:
            sys_cursors[r] = Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_N_RESIZE);
            break;
        case SDL_HITTEST_RESIZE_TOPRIGHT:
            sys_cursors[r] = Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_NE_RESIZE);
            break;
        case SDL_HITTEST_RESIZE_RIGHT:
            sys_cursors[r] = Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_E_RESIZE);
            break;
        case SDL_HITTEST_RESIZE_BOTTOMRIGHT:
            sys_cursors[r] = Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_SE_RESIZE);
            break;
        case SDL_HITTEST_RESIZE_BOTTOM:
            sys_cursors[r] = Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_S_RESIZE);
            break;
        case SDL_HITTEST_RESIZE_BOTTOMLEFT:
            sys_cursors[r] = Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_SW_RESIZE);
            break;
        case SDL_HITTEST_RESIZE_LEFT:
            sys_cursors[r] = Wayland_CreateSystemCursor(SDL_SYSTEM_CURSOR_W_RESIZE);
            break;
        }
        r++;
    }

#ifdef SDL_USE_LIBDBUS
    /* The DBus cursor properties are only needed when manually loading themes and cursors.
     * If the cursor shape protocol is present, the compositor will handle it internally.
     */
    if (!d->cursor_shape_manager) {
        Wayland_DBusInitCursorProperties(d);
    }
#endif

    SDL_SetDefaultCursor(Wayland_CreateDefaultCursor());
}

void Wayland_FiniMouse(SDL_VideoData *data)
{
    Wayland_FreeCursorThemes(data);

#ifdef SDL_USE_LIBDBUS
    Wayland_DBusFinishCursorProperties();
#endif

    for (int i = 0; i < SDL_arraysize(sys_cursors); i++) {
        Wayland_FreeCursor(sys_cursors[i]);
        sys_cursors[i] = NULL;
    }
}

void Wayland_SeatUpdateCursor(SDL_WaylandSeat *seat)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_WindowData *pointer_focus = seat->pointer.focus;

    if (pointer_focus) {
        const bool has_relative_focus = Wayland_SeatHasRelativePointerFocus(seat);

        if (!seat->display->relative_mode_enabled || !has_relative_focus || !mouse->relative_mode_hide_cursor) {
            const SDL_HitTestResult rc = pointer_focus->hit_test_result;

            if ((seat->display->relative_mode_enabled && has_relative_focus) ||
                rc == SDL_HITTEST_NORMAL || rc == SDL_HITTEST_DRAGGABLE) {
                Wayland_SeatSetCursor(seat, mouse->cur_cursor);
            } else {
                Wayland_SeatSetCursor(seat, sys_cursors[rc]);
            }
        } else {
            // Hide the cursor in relative mode, unless requested otherwise by the hint.
            Wayland_SeatSetCursor(seat, NULL);
        }
    } else {
        /* The spec states "The cursor actually changes only if the input device focus is one of the
         * requesting client's surfaces", so just clear the cursor if the seat has no pointer focus.
         */
        Wayland_SeatSetCursor(seat, NULL);
    }
}

#endif // SDL_VIDEO_DRIVER_WAYLAND
