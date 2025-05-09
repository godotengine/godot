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

#include "../../core/unix/SDL_poll.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_scancode_tables_c.h"
#include "../../events/SDL_keysym_to_keycode_c.h"
#include "../../core/linux/SDL_system_theme.h"
#include "../SDL_sysvideo.h"

#include "SDL_waylandvideo.h"
#include "SDL_waylandevents_c.h"
#include "SDL_waylandwindow.h"
#include "SDL_waylandmouse.h"
#include "SDL_waylandclipboard.h"

#include "pointer-constraints-unstable-v1-client-protocol.h"
#include "relative-pointer-unstable-v1-client-protocol.h"
#include "xdg-shell-client-protocol.h"
#include "keyboard-shortcuts-inhibit-unstable-v1-client-protocol.h"
#include "text-input-unstable-v3-client-protocol.h"
#include "tablet-v2-client-protocol.h"
#include "primary-selection-unstable-v1-client-protocol.h"
#include "input-timestamps-unstable-v1-client-protocol.h"

#ifdef HAVE_LIBDECOR_H
#include <libdecor.h>
#endif

#ifdef SDL_INPUT_LINUXEV
#include <linux/input.h>
#else
#define BTN_LEFT   (0x110)
#define BTN_RIGHT  (0x111)
#define BTN_MIDDLE (0x112)
#define BTN_SIDE   (0x113)
#define BTN_EXTRA  (0x114)
#endif
#include "../../events/SDL_keysym_to_scancode_c.h"
#include "../../events/imKStoUCS.h"
#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xkbcommon/xkbcommon-compose.h>
#include <xkbcommon/xkbcommon.h>
#include "cursor-shape-v1-client-protocol.h"
#include "viewporter-client-protocol.h"

// Weston uses a ratio of 10 units per scroll tick
#define WAYLAND_WHEEL_AXIS_UNIT 10

#ifndef XKB_MOD_NAME_MOD3
#define XKB_MOD_NAME_MOD3 "Mod3"
#endif

#ifndef XKB_MOD_NAME_MOD5
#define XKB_MOD_NAME_MOD5 "Mod5"
#endif

// Keyboard and mouse names to match XWayland
#define WAYLAND_DEFAULT_KEYBOARD_NAME "Virtual core keyboard"
#define WAYLAND_DEFAULT_POINTER_NAME "Virtual core pointer"
#define WAYLAND_DEFAULT_TOUCH_NAME "Virtual core touch"

// Focus clickthrough timeout
#define WAYLAND_FOCUS_CLICK_TIMEOUT_NS SDL_MS_TO_NS(10)

// Scoped function declarations
static void Wayland_SeatUpdateKeyboardGrab(SDL_WaylandSeat *seat);

struct SDL_WaylandTouchPoint
{
    SDL_TouchID id;
    wl_fixed_t fx;
    wl_fixed_t fy;
    struct wl_surface *surface;

    struct wl_list link;
};

static void Wayland_SeatAddTouch(SDL_WaylandSeat *seat, SDL_TouchID id, wl_fixed_t fx, wl_fixed_t fy, struct wl_surface *surface)
{
    struct SDL_WaylandTouchPoint *tp = SDL_malloc(sizeof(struct SDL_WaylandTouchPoint));

    SDL_zerop(tp);
    tp->id = id;
    tp->fx = fx;
    tp->fy = fy;
    tp->surface = surface;

    WAYLAND_wl_list_insert(&seat->touch.points, &tp->link);
}

static void Wayland_SeatUpdateTouch(SDL_WaylandSeat *seat, SDL_TouchID id, wl_fixed_t fx, wl_fixed_t fy, struct wl_surface **surface)
{
    struct SDL_WaylandTouchPoint *tp;

    wl_list_for_each (tp, &seat->touch.points, link) {
        if (tp->id == id) {
            tp->fx = fx;
            tp->fy = fy;
            if (surface) {
                *surface = tp->surface;
            }
            break;
        }
    }
}

static void Wayland_SeatRemoveTouch(SDL_WaylandSeat *seat, SDL_TouchID id, wl_fixed_t *fx, wl_fixed_t *fy, struct wl_surface **surface)
{
    struct SDL_WaylandTouchPoint *tp;

    wl_list_for_each (tp, &seat->touch.points, link) {
        if (tp->id == id) {
            if (fx) {
                *fx = tp->fx;
            }
            if (fy) {
                *fy = tp->fy;
            }
            if (surface) {
                *surface = tp->surface;
            }

            WAYLAND_wl_list_remove(&tp->link);
            SDL_free(tp);
            break;
        }
    }
}

static bool Wayland_SurfaceHasActiveTouches(SDL_VideoData *display, struct wl_surface *surface)
{
    struct SDL_WaylandTouchPoint *tp;
    SDL_WaylandSeat *seat;

    // Check all seats for active touches on the surface.
    wl_list_for_each (seat, &display->seat_list, link) {
        wl_list_for_each (tp, &seat->touch.points, link) {
            if (tp->surface == surface) {
                return true;
            }
        }
    }

    return false;
}

static void Wayland_GetScaledMouseRect(SDL_Window *window, SDL_Rect *scaled_rect)
{
    SDL_WindowData *window_data = window->internal;

    scaled_rect->x = (int)SDL_floor(window->mouse_rect.x / window_data->pointer_scale.x);
    scaled_rect->y = (int)SDL_floor(window->mouse_rect.y / window_data->pointer_scale.y);
    scaled_rect->w = (int)SDL_ceil(window->mouse_rect.w / window_data->pointer_scale.x);
    scaled_rect->h = (int)SDL_ceil(window->mouse_rect.h / window_data->pointer_scale.y);
}

static Uint64 Wayland_AdjustEventTimestampBase(Uint64 nsTimestamp)
{
    static Uint64 timestamp_offset = 0;
    const Uint64 now = SDL_GetTicksNS();

    if (!timestamp_offset) {
        timestamp_offset = (now - nsTimestamp);
    }
    nsTimestamp += timestamp_offset;

    if (nsTimestamp > now) {
        timestamp_offset -= (nsTimestamp - now);
        nsTimestamp = now;
    }

    return nsTimestamp;
}

/* This should only be called with 32-bit millisecond timestamps received in Wayland events!
 * No synthetic or high-res timestamps, as they can corrupt the rollover offset!
 */
static Uint64 Wayland_EventTimestampMSToNS(Uint32 wl_timestamp_ms)
{
    static Uint64 timestamp_offset = 0;
    static Uint32 last = 0;

    // Handle 32-bit timer rollover.
    if (wl_timestamp_ms < last) {
        timestamp_offset += SDL_MS_TO_NS(SDL_UINT64_C(0x100000000));
    }
    last = wl_timestamp_ms;
    wl_timestamp_ms += timestamp_offset;

    return SDL_MS_TO_NS(wl_timestamp_ms) + timestamp_offset;
}

/* Even if high-res timestamps are available, the millisecond timestamps are still processed
 * to accumulate the rollover offset if needed later.
 */

static Uint64 Wayland_GetKeyboardTimestampRaw(SDL_WaylandSeat *seat, Uint32 wl_timestamp_ms)
{
    const Uint64 adjustedTimestampMS = Wayland_EventTimestampMSToNS(wl_timestamp_ms);
    return seat->keyboard.timestamps ? seat->keyboard.highres_timestamp_ns : adjustedTimestampMS;
}

static Uint64 Wayland_GetPointerTimestamp(SDL_WaylandSeat *seat, Uint32 wl_timestamp_ms)
{
    const Uint64 adjustedTimestampMS = Wayland_EventTimestampMSToNS(wl_timestamp_ms);
    return Wayland_AdjustEventTimestampBase(seat->pointer.timestamps ? seat->pointer.highres_timestamp_ns : adjustedTimestampMS);
}

Uint64 Wayland_GetTouchTimestamp(SDL_WaylandSeat *seat, Uint32 wl_timestamp_ms)
{
    const Uint64 adjustedTimestampMS = Wayland_EventTimestampMSToNS(wl_timestamp_ms);
    return Wayland_AdjustEventTimestampBase(seat->touch.timestamps ? seat->touch.highres_timestamp_ns : adjustedTimestampMS);
}

static void input_timestamp_listener(void *data, struct zwp_input_timestamps_v1 *zwp_input_timestamps_v1,
                                             uint32_t tv_sec_hi, uint32_t tv_sec_lo, uint32_t tv_nsec)
{
    *((Uint64 *)data) = ((((Uint64)tv_sec_hi << 32) | (Uint64)tv_sec_lo) * SDL_NS_PER_SECOND) + tv_nsec;
}

static const struct zwp_input_timestamps_v1_listener timestamp_listener = {
    input_timestamp_listener
};

static void Wayland_SeatRegisterInputTimestampListeners(SDL_WaylandSeat *seat)
{
    if (seat->display->input_timestamps_manager) {
        if (seat->keyboard.wl_keyboard && !seat->keyboard.timestamps) {
            seat->keyboard.timestamps = zwp_input_timestamps_manager_v1_get_keyboard_timestamps(seat->display->input_timestamps_manager, seat->keyboard.wl_keyboard);
            zwp_input_timestamps_v1_add_listener(seat->keyboard.timestamps, &timestamp_listener, &seat->keyboard.highres_timestamp_ns);
        }

        if (seat->pointer.wl_pointer && !seat->pointer.timestamps) {
            seat->pointer.timestamps = zwp_input_timestamps_manager_v1_get_pointer_timestamps(seat->display->input_timestamps_manager, seat->pointer.wl_pointer);
            zwp_input_timestamps_v1_add_listener(seat->pointer.timestamps, &timestamp_listener, &seat->pointer.highres_timestamp_ns);
        }

        if (seat->touch.wl_touch && !seat->touch.timestamps) {
            seat->touch.timestamps = zwp_input_timestamps_manager_v1_get_touch_timestamps(seat->display->input_timestamps_manager, seat->touch.wl_touch);
            zwp_input_timestamps_v1_add_listener(seat->touch.timestamps, &timestamp_listener, &seat->touch.highres_timestamp_ns);
        }
    }
}

void Wayland_DisplayInitInputTimestampManager(SDL_VideoData *display)
{
    if (display->input_timestamps_manager) {
        SDL_WaylandSeat *seat;
        wl_list_for_each (seat, &display->seat_list, link) {
            Wayland_SeatRegisterInputTimestampListeners(seat);
        }
    }
}

static void Wayland_SeatCreateCursorShape(SDL_WaylandSeat *seat)
{
    if (seat->display->cursor_shape_manager) {
        if (seat->pointer.wl_pointer && !seat->pointer.cursor_shape) {
            seat->pointer.cursor_shape = wp_cursor_shape_manager_v1_get_pointer(seat->display->cursor_shape_manager, seat->pointer.wl_pointer);
        }
    }
}

void Wayland_DisplayInitCursorShapeManager(SDL_VideoData *display)
{
    SDL_WaylandSeat *seat;
    wl_list_for_each (seat, &display->seat_list, link) {
        Wayland_SeatCreateCursorShape(seat);
    }
}

// Returns true if a key repeat event was due
static bool keyboard_repeat_handle(SDL_WaylandKeyboardRepeat *repeat_info, Uint64 elapsed)
{
    bool ret = false;
    while (elapsed >= repeat_info->next_repeat_ns) {
        if (repeat_info->scancode != SDL_SCANCODE_UNKNOWN) {
            const Uint64 timestamp = repeat_info->wl_press_time_ns + repeat_info->next_repeat_ns;
            SDL_SendKeyboardKeyIgnoreModifiers(Wayland_AdjustEventTimestampBase(timestamp), repeat_info->keyboard_id, repeat_info->key, repeat_info->scancode, true);
        }
        if (repeat_info->text[0]) {
            SDL_SendKeyboardText(repeat_info->text);
        }
        repeat_info->next_repeat_ns += SDL_NS_PER_SECOND / (Uint64)repeat_info->repeat_rate;
        ret = true;
    }
    return ret;
}

static void keyboard_repeat_clear(SDL_WaylandKeyboardRepeat *repeat_info)
{
    if (!repeat_info->is_initialized) {
        return;
    }
    repeat_info->is_key_down = false;
}

static void keyboard_repeat_set(SDL_WaylandKeyboardRepeat *repeat_info, Uint32 keyboard_id, uint32_t key, Uint64 wl_press_time_ns,
                                uint32_t scancode, bool has_text, char text[8])
{
    if (!repeat_info->is_initialized || !repeat_info->repeat_rate) {
        return;
    }
    repeat_info->is_key_down = true;
    repeat_info->keyboard_id = keyboard_id;
    repeat_info->key = key;
    repeat_info->wl_press_time_ns = wl_press_time_ns;
    repeat_info->sdl_press_time_ns = SDL_GetTicksNS();
    repeat_info->next_repeat_ns = SDL_MS_TO_NS(repeat_info->repeat_delay_ms);
    repeat_info->scancode = scancode;
    if (has_text) {
        SDL_copyp(repeat_info->text, text);
    } else {
        repeat_info->text[0] = '\0';
    }
}

static uint32_t keyboard_repeat_get_key(SDL_WaylandKeyboardRepeat *repeat_info)
{
    if (repeat_info->is_initialized && repeat_info->is_key_down) {
        return repeat_info->key;
    }

    return 0;
}

static void keyboard_repeat_set_text(SDL_WaylandKeyboardRepeat *repeat_info, const char text[8])
{
    if (repeat_info->is_initialized) {
        SDL_copyp(repeat_info->text, text);
    }
}

static bool keyboard_repeat_is_set(SDL_WaylandKeyboardRepeat *repeat_info)
{
    return repeat_info->is_initialized && repeat_info->is_key_down;
}

static bool keyboard_repeat_key_is_set(SDL_WaylandKeyboardRepeat *repeat_info, uint32_t key)
{
    return repeat_info->is_initialized && repeat_info->is_key_down && key == repeat_info->key;
}

static void sync_done_handler(void *data, struct wl_callback *callback, uint32_t callback_data)
{
    // Nothing to do, just destroy the callback
    wl_callback_destroy(callback);
}

static struct wl_callback_listener sync_listener = {
    sync_done_handler
};

void Wayland_SendWakeupEvent(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData *d = _this->internal;

    /* Queue a sync event to unblock the event queue fd if it's empty and being waited on.
     * TODO: Maybe use a pipe to avoid the compositor roundtrip?
     */
    struct wl_callback *cb = wl_display_sync(d->display);
    wl_callback_add_listener(cb, &sync_listener, NULL);
    WAYLAND_wl_display_flush(d->display);
}

static int dispatch_queued_events(SDL_VideoData *viddata)
{
    int rc;

    /*
     * NOTE: When reconnection is implemented, check if libdecor needs to be
     *       involved in the reconnection process.
     */
#ifdef HAVE_LIBDECOR_H
    if (viddata->shell.libdecor) {
        libdecor_dispatch(viddata->shell.libdecor, 0);
    }
#endif

    rc = WAYLAND_wl_display_dispatch_pending(viddata->display);
    return rc >= 0 ? 1 : rc;
}

int Wayland_WaitEventTimeout(SDL_VideoDevice *_this, Sint64 timeoutNS)
{
    SDL_VideoData *d = _this->internal;
    SDL_WaylandSeat *seat;
    bool key_repeat_active = false;

    WAYLAND_wl_display_flush(d->display);

#ifdef SDL_USE_IME
    SDL_Window *keyboard_focus = SDL_GetKeyboardFocus();
    if (!d->text_input_manager && keyboard_focus && SDL_TextInputActive(keyboard_focus)) {
        SDL_IME_PumpEvents();
    }
#endif

#ifdef SDL_USE_LIBDBUS
    SDL_DBus_PumpEvents();
#endif

    // If key repeat is active, we'll need to cap our maximum wait time to handle repeats
    wl_list_for_each (seat, &d->seat_list, link) {
        if (keyboard_repeat_is_set(&seat->keyboard.repeat)) {
            if (seat->keyboard.sdl_keymap != SDL_GetCurrentKeymap()) {
                SDL_SetKeymap(seat->keyboard.sdl_keymap, true);
                SDL_SetModState(seat->keyboard.pressed_modifiers | seat->keyboard.locked_modifiers);
            }

            const Uint64 elapsed = SDL_GetTicksNS() - seat->keyboard.repeat.sdl_press_time_ns;
            if (keyboard_repeat_handle(&seat->keyboard.repeat, elapsed)) {
                // A repeat key event was already due
                return 1;
            } else {
                const Uint64 next_repeat_wait_time = (seat->keyboard.repeat.next_repeat_ns - elapsed) + 1;
                if (timeoutNS >= 0) {
                    timeoutNS = SDL_min(timeoutNS, next_repeat_wait_time);
                } else {
                    timeoutNS = next_repeat_wait_time;
                }
                key_repeat_active = true;
            }
        }
    }

    /* wl_display_prepare_read() will return -1 if the default queue is not empty.
     * If the default queue is empty, it will prepare us for our SDL_IOReady() call. */
    if (WAYLAND_wl_display_prepare_read(d->display) == 0) {
        // Use SDL_IOR_NO_RETRY to ensure SIGINT will break us out of our wait
        int err = SDL_IOReady(WAYLAND_wl_display_get_fd(d->display), SDL_IOR_READ | SDL_IOR_NO_RETRY, timeoutNS);
        if (err > 0) {
            // There are new events available to read
            WAYLAND_wl_display_read_events(d->display);
            return dispatch_queued_events(d);
        } else if (err == 0) {
            int ret = 0;

            // No events available within the timeout
            WAYLAND_wl_display_cancel_read(d->display);

            // If key repeat is active, we might have woken up to generate a key event
            if (key_repeat_active) {
                wl_list_for_each (seat, &d->seat_list, link) {
                    if (seat->keyboard.sdl_keymap != SDL_GetCurrentKeymap()) {
                        SDL_SetKeymap(seat->keyboard.sdl_keymap, true);
                        SDL_SetModState(seat->keyboard.pressed_modifiers | seat->keyboard.locked_modifiers);
                    }

                    const Uint64 elapsed = SDL_GetTicksNS() - seat->keyboard.repeat.sdl_press_time_ns;
                    if (keyboard_repeat_handle(&seat->keyboard.repeat, elapsed)) {
                        ++ret;
                    }
                }
            }

            return ret;
        } else {
            // Error returned from poll()/select()
            WAYLAND_wl_display_cancel_read(d->display);

            if (errno == EINTR) {
                /* If the wait was interrupted by a signal, we may have generated a
                 * SDL_EVENT_QUIT event. Let the caller know to call SDL_PumpEvents(). */
                return 1;
            } else {
                return err;
            }
        }
    } else {
        // We already had pending events
        return dispatch_queued_events(d);
    }
}

void Wayland_PumpEvents(SDL_VideoDevice *_this)
{
    SDL_VideoData *d = _this->internal;
    SDL_WaylandSeat *seat;
    int err;

#ifdef SDL_USE_IME
    SDL_Window *keyboard_focus = SDL_GetKeyboardFocus();
    if (!d->text_input_manager && keyboard_focus && SDL_TextInputActive(keyboard_focus)) {
        SDL_IME_PumpEvents();
    }
#endif

#ifdef SDL_USE_LIBDBUS
    SDL_DBus_PumpEvents();
#endif

#ifdef HAVE_LIBDECOR_H
    if (d->shell.libdecor) {
        libdecor_dispatch(d->shell.libdecor, 0);
    }
#endif

    WAYLAND_wl_display_flush(d->display);

    /* wl_display_prepare_read() will return -1 if the default queue is not empty.
     * If the default queue is empty, it will prepare us for our SDL_IOReady() call. */
    if (WAYLAND_wl_display_prepare_read(d->display) == 0) {
        if (SDL_IOReady(WAYLAND_wl_display_get_fd(d->display), SDL_IOR_READ, 0) > 0) {
            WAYLAND_wl_display_read_events(d->display);
        } else {
            WAYLAND_wl_display_cancel_read(d->display);
        }
    }

    // Dispatch any pre-existing pending events or new events we may have read
    err = WAYLAND_wl_display_dispatch_pending(d->display);

    wl_list_for_each (seat, &d->seat_list, link) {
        if (keyboard_repeat_is_set(&seat->keyboard.repeat)) {
            if (seat->keyboard.sdl_keymap != SDL_GetCurrentKeymap()) {
                SDL_SetKeymap(seat->keyboard.sdl_keymap, true);
                SDL_SetModState(seat->keyboard.pressed_modifiers | seat->keyboard.locked_modifiers);
            }

            const Uint64 elapsed = SDL_GetTicksNS() - seat->keyboard.repeat.sdl_press_time_ns;
            keyboard_repeat_handle(&seat->keyboard.repeat, elapsed);
        }
    }

    if (err < 0 && !d->display_disconnected) {
        /* Something has failed with the Wayland connection -- for example,
         * the compositor may have shut down and closed its end of the socket,
         * or there is a library-specific error.
         *
         * Try to recover once, then quit.
         */
        if (!Wayland_VideoReconnect(_this)) {
            d->display_disconnected = 1;
            SDL_LogError(SDL_LOG_CATEGORY_VIDEO, "Wayland display connection closed by server (fatal)");

            /* Only send a single quit message, as application shutdown might call
             * SDL_PumpEvents
             */
            SDL_SendQuit();
        }
    }
}

static void pointer_handle_motion_common(SDL_WaylandSeat *seat, Uint64 nsTimestamp, wl_fixed_t sx_w, wl_fixed_t sy_w)
{
    SDL_WindowData *window_data = seat->pointer.focus;
    SDL_Window *window = window_data ? window_data->sdlwindow : NULL;

    if (window_data) {
        const float sx = (float)(wl_fixed_to_double(sx_w) * window_data->pointer_scale.x);
        const float sy = (float)(wl_fixed_to_double(sy_w) * window_data->pointer_scale.y);
        SDL_SendMouseMotion(nsTimestamp, window_data->sdlwindow, seat->pointer.sdl_id, false, sx, sy);

        seat->pointer.last_motion.x = (int)SDL_floorf(sx);
        seat->pointer.last_motion.y = (int)SDL_floorf(sy);

        /* Pointer confinement regions are created only when the pointer actually enters the region via
         * a motion event received from the compositor.
         */
        if (!SDL_RectEmpty(&window->mouse_rect) && !seat->pointer.confined_pointer) {
            SDL_Rect scaled_mouse_rect;
            Wayland_GetScaledMouseRect(window, &scaled_mouse_rect);

            if (SDL_PointInRect(&seat->pointer.last_motion, &scaled_mouse_rect)) {
                Wayland_SeatUpdatePointerGrab(seat);
            }
        }

        if (window->hit_test) {
            SDL_HitTestResult rc = window->hit_test(window, &seat->pointer.last_motion, window->hit_test_data);

            // Apply the toplevel constraints if the window isn't resizable from those directions.
            switch (rc) {
            case SDL_HITTEST_RESIZE_TOPLEFT:
                if ((window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_TOP) &&
                    (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_LEFT)) {
                    rc = SDL_HITTEST_NORMAL;
                } else if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_TOP) {
                    rc = SDL_HITTEST_RESIZE_LEFT;
                } else if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_LEFT) {
                    rc = SDL_HITTEST_RESIZE_TOP;
                }
                break;
            case SDL_HITTEST_RESIZE_TOP:
                if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_TOP) {
                    rc = SDL_HITTEST_NORMAL;
                }
                break;
            case SDL_HITTEST_RESIZE_TOPRIGHT:
                if ((window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_TOP) &&
                    (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_RIGHT)) {
                    rc = SDL_HITTEST_NORMAL;
                } else if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_TOP) {
                    rc = SDL_HITTEST_RESIZE_RIGHT;
                } else if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_RIGHT) {
                    rc = SDL_HITTEST_RESIZE_TOP;
                }
                break;
            case SDL_HITTEST_RESIZE_RIGHT:
                if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_RIGHT) {
                    rc = SDL_HITTEST_NORMAL;
                }
                break;
            case SDL_HITTEST_RESIZE_BOTTOMRIGHT:
                if ((window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_BOTTOM) &&
                    (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_RIGHT)) {
                    rc = SDL_HITTEST_NORMAL;
                } else if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_BOTTOM) {
                    rc = SDL_HITTEST_RESIZE_RIGHT;
                } else if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_RIGHT) {
                    rc = SDL_HITTEST_RESIZE_BOTTOM;
                }
                break;
            case SDL_HITTEST_RESIZE_BOTTOM:
                if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_BOTTOM) {
                    rc = SDL_HITTEST_NORMAL;
                }
                break;
            case SDL_HITTEST_RESIZE_BOTTOMLEFT:
                if ((window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_BOTTOM) &&
                    (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_LEFT)) {
                    rc = SDL_HITTEST_NORMAL;
                } else if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_BOTTOM) {
                    rc = SDL_HITTEST_RESIZE_LEFT;
                } else if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_LEFT) {
                    rc = SDL_HITTEST_RESIZE_BOTTOM;
                }
                break;
            case SDL_HITTEST_RESIZE_LEFT:
                if (window_data->toplevel_constraints & WAYLAND_TOPLEVEL_CONSTRAINED_LEFT) {
                    rc = SDL_HITTEST_NORMAL;
                }
                break;
            default:
                break;
            }

            if (rc != window_data->hit_test_result) {
                window_data->hit_test_result = rc;
                Wayland_SeatUpdateCursor(seat);
            }
        }
    }
}

static void pointer_handle_motion(void *data, struct wl_pointer *pointer,
                                  uint32_t time, wl_fixed_t sx_w, wl_fixed_t sy_w)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    pointer_handle_motion_common(seat, Wayland_GetPointerTimestamp(seat, time), sx_w, sy_w);
}

static void pointer_handle_enter(void *data, struct wl_pointer *pointer,
                                 uint32_t serial, struct wl_surface *surface,
                                 wl_fixed_t sx_w, wl_fixed_t sy_w)
{
    if (!surface) {
        // Enter event for a destroyed surface.
        return;
    }

    SDL_WindowData *window = Wayland_GetWindowDataForOwnedSurface(surface);
    if (!window) {
        // Not a surface owned by SDL.
        return;
    }

    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    seat->pointer.focus = window;
    seat->pointer.enter_serial = serial;
    ++window->pointer_focus_count;
    SDL_SetMouseFocus(window->sdlwindow);

    /* In the case of e.g. a pointer confine warp, we may receive an enter
     * event with no following motion event, but with the new coordinates
     * as part of the enter event.
     *
     * FIXME: This causes a movement event with an anomalous timestamp when
     *        the cursor enters the window.
     */
    pointer_handle_motion_common(seat, 0, sx_w, sy_w);

    // Update the pointer grab state.
    Wayland_SeatUpdatePointerGrab(seat);

    /* If the cursor was changed while our window didn't have pointer
     * focus, we might need to trigger another call to
     * wl_pointer_set_cursor() for the new cursor to be displayed.
     *
     * This will also update the cursor if a second pointer entered a
     * window that already has focus, as the focus change sequence
     * won't be run.
     */
    Wayland_SeatUpdateCursor(seat);
}

static void pointer_handle_leave(void *data, struct wl_pointer *pointer,
                                 uint32_t serial, struct wl_surface *surface)
{
    if (!surface) {
        // Leave event for a destroyed surface.
        return;
    }

    SDL_WindowData *window = Wayland_GetWindowDataForOwnedSurface(surface);
    if (!window) {
        // Not a surface owned by SDL.
        return;
    }

    // Clear the capture flag and raise all buttons
    window->sdlwindow->flags &= ~SDL_WINDOW_MOUSE_CAPTURE;

    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    seat->pointer.focus = NULL;
    seat->pointer.buttons_pressed = 0;
    SDL_SendMouseButton(0, window->sdlwindow, seat->pointer.sdl_id, SDL_BUTTON_LEFT, false);
    SDL_SendMouseButton(0, window->sdlwindow, seat->pointer.sdl_id, SDL_BUTTON_RIGHT, false);
    SDL_SendMouseButton(0, window->sdlwindow, seat->pointer.sdl_id, SDL_BUTTON_MIDDLE, false);
    SDL_SendMouseButton(0, window->sdlwindow, seat->pointer.sdl_id, SDL_BUTTON_X1, false);
    SDL_SendMouseButton(0, window->sdlwindow, seat->pointer.sdl_id, SDL_BUTTON_X2, false);

    /* A pointer leave event may be emitted if the compositor hides the pointer in response to receiving a touch event.
     * Don't relinquish focus if the surface has active touches, as the compositor is just transitioning from mouse to touch mode.
     */
    SDL_Window *mouse_focus = SDL_GetMouseFocus();
    const bool had_focus = mouse_focus && window->sdlwindow == mouse_focus;
    if (!--window->pointer_focus_count && had_focus && !Wayland_SurfaceHasActiveTouches(seat->display, surface)) {
        SDL_SetMouseFocus(NULL);
    }

    Wayland_SeatUpdatePointerGrab(seat);
    Wayland_SeatUpdateCursor(seat);
}

static bool Wayland_ProcessHitTest(SDL_WaylandSeat *seat, Uint32 serial)
{
    // Locked in relative mode, do nothing.
    if (seat->pointer.locked_pointer) {
        return false;
    }

    SDL_WindowData *window_data = seat->pointer.focus;
    SDL_Window *window = window_data->sdlwindow;

    if (window->hit_test) {
        static const uint32_t directions[] = {
            XDG_TOPLEVEL_RESIZE_EDGE_TOP_LEFT, XDG_TOPLEVEL_RESIZE_EDGE_TOP,
            XDG_TOPLEVEL_RESIZE_EDGE_TOP_RIGHT, XDG_TOPLEVEL_RESIZE_EDGE_RIGHT,
            XDG_TOPLEVEL_RESIZE_EDGE_BOTTOM_RIGHT, XDG_TOPLEVEL_RESIZE_EDGE_BOTTOM,
            XDG_TOPLEVEL_RESIZE_EDGE_BOTTOM_LEFT, XDG_TOPLEVEL_RESIZE_EDGE_LEFT
        };

#ifdef HAVE_LIBDECOR_H
        static const uint32_t directions_libdecor[] = {
            LIBDECOR_RESIZE_EDGE_TOP_LEFT, LIBDECOR_RESIZE_EDGE_TOP,
            LIBDECOR_RESIZE_EDGE_TOP_RIGHT, LIBDECOR_RESIZE_EDGE_RIGHT,
            LIBDECOR_RESIZE_EDGE_BOTTOM_RIGHT, LIBDECOR_RESIZE_EDGE_BOTTOM,
            LIBDECOR_RESIZE_EDGE_BOTTOM_LEFT, LIBDECOR_RESIZE_EDGE_LEFT
        };
#endif

        switch (window_data->hit_test_result) {
        case SDL_HITTEST_DRAGGABLE:
#ifdef HAVE_LIBDECOR_H
            if (window_data->shell_surface_type == WAYLAND_SHELL_SURFACE_TYPE_LIBDECOR) {
                if (window_data->shell_surface.libdecor.frame) {
                    libdecor_frame_move(window_data->shell_surface.libdecor.frame,
                                        seat->wl_seat,
                                        serial);
                }
            } else
#endif
                if (window_data->shell_surface_type == WAYLAND_SHELL_SURFACE_TYPE_XDG_TOPLEVEL) {
                if (window_data->shell_surface.xdg.toplevel.xdg_toplevel) {
                    xdg_toplevel_move(window_data->shell_surface.xdg.toplevel.xdg_toplevel,
                                      seat->wl_seat,
                                      serial);
                }
            }
            return true;

        case SDL_HITTEST_RESIZE_TOPLEFT:
        case SDL_HITTEST_RESIZE_TOP:
        case SDL_HITTEST_RESIZE_TOPRIGHT:
        case SDL_HITTEST_RESIZE_RIGHT:
        case SDL_HITTEST_RESIZE_BOTTOMRIGHT:
        case SDL_HITTEST_RESIZE_BOTTOM:
        case SDL_HITTEST_RESIZE_BOTTOMLEFT:
        case SDL_HITTEST_RESIZE_LEFT:
#ifdef HAVE_LIBDECOR_H
            if (window_data->shell_surface_type == WAYLAND_SHELL_SURFACE_TYPE_LIBDECOR) {
                if (window_data->shell_surface.libdecor.frame) {
                    libdecor_frame_resize(window_data->shell_surface.libdecor.frame,
                                          seat->wl_seat,
                                          serial,
                                          directions_libdecor[window_data->hit_test_result - SDL_HITTEST_RESIZE_TOPLEFT]);
                }
            } else
#endif
                if (window_data->shell_surface_type == WAYLAND_SHELL_SURFACE_TYPE_XDG_TOPLEVEL) {
                if (window_data->shell_surface.xdg.toplevel.xdg_toplevel) {
                    xdg_toplevel_resize(window_data->shell_surface.xdg.toplevel.xdg_toplevel,
                                        seat->wl_seat,
                                        serial,
                                        directions[window_data->hit_test_result - SDL_HITTEST_RESIZE_TOPLEFT]);
                }
            }
            return true;

        default:
            return false;
        }
    }

    return false;
}

static void pointer_handle_button_common(SDL_WaylandSeat *seat, uint32_t serial,
                                         Uint64 nsTimestamp, uint32_t button, uint32_t state_w)
{
    SDL_WindowData *window = seat->pointer.focus;
    enum wl_pointer_button_state state = state_w;
    Uint8 sdl_button;
    const bool down = (state != 0);

    switch (button) {
    case BTN_LEFT:
        sdl_button = SDL_BUTTON_LEFT;
        break;
    case BTN_MIDDLE:
        sdl_button = SDL_BUTTON_MIDDLE;
        break;
    case BTN_RIGHT:
        sdl_button = SDL_BUTTON_RIGHT;
        break;
    case BTN_SIDE:
        sdl_button = SDL_BUTTON_X1;
        break;
    case BTN_EXTRA:
        sdl_button = SDL_BUTTON_X2;
        break;
    default:
        return;
    }

    if (window) {
        bool ignore_click = false;

        if (state) {
            Wayland_UpdateImplicitGrabSerial(seat, serial);
            seat->pointer.buttons_pressed |= SDL_BUTTON_MASK(sdl_button);
        } else {
            seat->pointer.buttons_pressed &= ~(SDL_BUTTON_MASK(sdl_button));
        }

        if (sdl_button == SDL_BUTTON_LEFT && Wayland_ProcessHitTest(seat, serial)) {
            return; // don't pass this event on to app.
        }

        // Possibly ignore this click if it was to gain focus.
        if (window->last_focus_event_time_ns) {
            if (state == WL_POINTER_BUTTON_STATE_PRESSED &&
                (SDL_GetTicksNS() - window->last_focus_event_time_ns) < WAYLAND_FOCUS_CLICK_TIMEOUT_NS) {
                ignore_click = !SDL_GetHintBoolean(SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, false);
            }

            window->last_focus_event_time_ns = 0;
        }

        /* Wayland won't let you "capture" the mouse, but it will automatically track
         * the mouse outside the window if you drag outside of it, until you let go
         * of all buttons (even if you add or remove presses outside the window, as
         * long as any button is still down, the capture remains).
         *
         * The mouse is not captured in relative mode.
         */
        if (!seat->display->relative_mode_enabled || !Wayland_SeatHasRelativePointerFocus(seat)) {
            if (seat->pointer.buttons_pressed != 0) {
                window->sdlwindow->flags |= SDL_WINDOW_MOUSE_CAPTURE;
            } else {
                window->sdlwindow->flags &= ~SDL_WINDOW_MOUSE_CAPTURE;
            }
        }

        if (!ignore_click) {
            SDL_SendMouseButton(nsTimestamp, window->sdlwindow, seat->pointer.sdl_id, sdl_button, down);
        }
    }
}

static void pointer_handle_button(void *data, struct wl_pointer *pointer, uint32_t serial,
                                  uint32_t time, uint32_t button, uint32_t state_w)
{
    SDL_WaylandSeat *seat = data;
    pointer_handle_button_common(seat, serial, Wayland_GetPointerTimestamp(seat, time), button, state_w);
}

static void pointer_handle_axis_common_v1(SDL_WaylandSeat *seat,
                                          Uint64 nsTimestamp, uint32_t axis, wl_fixed_t value)
{
    SDL_WindowData *window = seat->pointer.focus;
    const enum wl_pointer_axis a = axis;

    if (seat->pointer.focus) {
        float x, y;

        switch (a) {
        case WL_POINTER_AXIS_VERTICAL_SCROLL:
            x = 0;
            y = 0 - (float)wl_fixed_to_double(value);
            break;
        case WL_POINTER_AXIS_HORIZONTAL_SCROLL:
            x = (float)wl_fixed_to_double(value);
            y = 0;
            break;
        default:
            return;
        }

        x /= WAYLAND_WHEEL_AXIS_UNIT;
        y /= WAYLAND_WHEEL_AXIS_UNIT;

        SDL_SendMouseWheel(nsTimestamp, window->sdlwindow, seat->pointer.sdl_id, x, y, SDL_MOUSEWHEEL_NORMAL);
    }
}

static void pointer_handle_axis_common(SDL_WaylandSeat *seat, enum SDL_WaylandAxisEvent type,
                                       uint32_t axis, wl_fixed_t value)
{
    const enum wl_pointer_axis a = axis;

    if (seat->pointer.focus) {
        switch (a) {
        case WL_POINTER_AXIS_VERTICAL_SCROLL:
            switch (type) {
            case AXIS_EVENT_VALUE120:
                /*
                 * High resolution scroll event. The spec doesn't state that axis_value120
                 * events are limited to one per frame, so the values are accumulated.
                 */
                if (seat->pointer.current_axis_info.y_axis_type != AXIS_EVENT_VALUE120) {
                    seat->pointer.current_axis_info.y_axis_type = AXIS_EVENT_VALUE120;
                    seat->pointer.current_axis_info.y = 0.0f;
                }
                seat->pointer.current_axis_info.y += 0 - (float)wl_fixed_to_double(value);
                break;
            case AXIS_EVENT_DISCRETE:
                /*
                 * This is a discrete axis event, so we process it and set the
                 * flag to ignore future continuous axis events in this frame.
                 */
                if (seat->pointer.current_axis_info.y_axis_type != AXIS_EVENT_DISCRETE) {
                    seat->pointer.current_axis_info.y_axis_type = AXIS_EVENT_DISCRETE;
                    seat->pointer.current_axis_info.y = 0 - (float)wl_fixed_to_double(value);
                }
                break;
            case AXIS_EVENT_CONTINUOUS:
                // Only process continuous events if no discrete events have been received.
                if (seat->pointer.current_axis_info.y_axis_type == AXIS_EVENT_CONTINUOUS) {
                    seat->pointer.current_axis_info.y = 0 - (float)wl_fixed_to_double(value);
                }
                break;
            }
            break;
        case WL_POINTER_AXIS_HORIZONTAL_SCROLL:
            switch (type) {
            case AXIS_EVENT_VALUE120:
                /*
                 * High resolution scroll event. The spec doesn't state that axis_value120
                 * events are limited to one per frame, so the values are accumulated.
                 */
                if (seat->pointer.current_axis_info.x_axis_type != AXIS_EVENT_VALUE120) {
                    seat->pointer.current_axis_info.x_axis_type = AXIS_EVENT_VALUE120;
                    seat->pointer.current_axis_info.x = 0.0f;
                }
                seat->pointer.current_axis_info.x += (float)wl_fixed_to_double(value);
                break;
            case AXIS_EVENT_DISCRETE:
                /*
                 * This is a discrete axis event, so we process it and set the
                 * flag to ignore future continuous axis events in this frame.
                 */
                if (seat->pointer.current_axis_info.x_axis_type != AXIS_EVENT_DISCRETE) {
                    seat->pointer.current_axis_info.x_axis_type = AXIS_EVENT_DISCRETE;
                    seat->pointer.current_axis_info.x = (float)wl_fixed_to_double(value);
                }
                break;
            case AXIS_EVENT_CONTINUOUS:
                // Only process continuous events if no discrete events have been received.
                if (seat->pointer.current_axis_info.x_axis_type == AXIS_EVENT_CONTINUOUS) {
                    seat->pointer.current_axis_info.x = (float)wl_fixed_to_double(value);
                }
                break;
            }
            break;
        }
    }
}

static void pointer_handle_axis(void *data, struct wl_pointer *pointer,
                                uint32_t time, uint32_t axis, wl_fixed_t value)
{
    SDL_WaylandSeat *seat = data;
    const Uint64 nsTimestamp = Wayland_GetPointerTimestamp(seat, time);

    if (wl_seat_get_version(seat->wl_seat) >= WL_POINTER_FRAME_SINCE_VERSION) {
        seat->pointer.current_axis_info.timestamp_ns = nsTimestamp;
        pointer_handle_axis_common(seat, AXIS_EVENT_CONTINUOUS, axis, value);
    } else {
        pointer_handle_axis_common_v1(seat, nsTimestamp, axis, value);
    }
}

static void pointer_handle_axis_relative_direction(void *data, struct wl_pointer *pointer,
                                                   uint32_t axis, uint32_t axis_relative_direction)
{
    SDL_WaylandSeat *seat = data;
    if (axis != WL_POINTER_AXIS_VERTICAL_SCROLL) {
        return;
    }
    switch (axis_relative_direction) {
    case WL_POINTER_AXIS_RELATIVE_DIRECTION_IDENTICAL:
        seat->pointer.current_axis_info.direction = SDL_MOUSEWHEEL_NORMAL;
        break;
    case WL_POINTER_AXIS_RELATIVE_DIRECTION_INVERTED:
        seat->pointer.current_axis_info.direction = SDL_MOUSEWHEEL_FLIPPED;
        break;
    }
}

static void pointer_handle_frame(void *data, struct wl_pointer *pointer)
{
    SDL_WaylandSeat *seat = data;
    SDL_WindowData *window = seat->pointer.focus;
    float x, y;
    SDL_MouseWheelDirection direction = seat->pointer.current_axis_info.direction;

    switch (seat->pointer.current_axis_info.x_axis_type) {
    case AXIS_EVENT_CONTINUOUS:
        x = seat->pointer.current_axis_info.x / WAYLAND_WHEEL_AXIS_UNIT;
        break;
    case AXIS_EVENT_DISCRETE:
        x = seat->pointer.current_axis_info.x;
        break;
    case AXIS_EVENT_VALUE120:
        x = seat->pointer.current_axis_info.x / 120.0f;
        break;
    default:
        x = 0.0f;
        break;
    }

    switch (seat->pointer.current_axis_info.y_axis_type) {
    case AXIS_EVENT_CONTINUOUS:
        y = seat->pointer.current_axis_info.y / WAYLAND_WHEEL_AXIS_UNIT;
        break;
    case AXIS_EVENT_DISCRETE:
        y = seat->pointer.current_axis_info.y;
        break;
    case AXIS_EVENT_VALUE120:
        y = seat->pointer.current_axis_info.y / 120.0f;
        break;
    default:
        y = 0.0f;
        break;
    }

    // clear pointer.current_axis_info for next frame
    SDL_memset(&seat->pointer.current_axis_info, 0, sizeof(seat->pointer.current_axis_info));

    if (x != 0.0f || y != 0.0f) {
        SDL_SendMouseWheel(seat->pointer.current_axis_info.timestamp_ns,
                           window->sdlwindow, seat->pointer.sdl_id, x, y, direction);
    }
}

static void pointer_handle_axis_source(void *data, struct wl_pointer *pointer,
                                       uint32_t axis_source)
{
    // unimplemented
}

static void pointer_handle_axis_stop(void *data, struct wl_pointer *pointer,
                                     uint32_t time, uint32_t axis)
{
    // unimplemented
}

static void pointer_handle_axis_discrete(void *data, struct wl_pointer *pointer,
                                         uint32_t axis, int32_t discrete)
{
    SDL_WaylandSeat *seat = data;

    pointer_handle_axis_common(seat, AXIS_EVENT_DISCRETE, axis, wl_fixed_from_int(discrete));
}

static void pointer_handle_axis_value120(void *data, struct wl_pointer *pointer,
                                         uint32_t axis, int32_t value120)
{
    SDL_WaylandSeat *seat = data;

    pointer_handle_axis_common(seat, AXIS_EVENT_VALUE120, axis, wl_fixed_from_int(value120));
}

static const struct wl_pointer_listener pointer_listener = {
    pointer_handle_enter,
    pointer_handle_leave,
    pointer_handle_motion,
    pointer_handle_button,
    pointer_handle_axis,
    pointer_handle_frame,                  // Version 5
    pointer_handle_axis_source,            // Version 5
    pointer_handle_axis_stop,              // Version 5
    pointer_handle_axis_discrete,          // Version 5
    pointer_handle_axis_value120,          // Version 8
    pointer_handle_axis_relative_direction // Version 9
};

static void relative_pointer_handle_relative_motion(void *data,
                                                    struct zwp_relative_pointer_v1 *pointer,
                                                    uint32_t time_hi,
                                                    uint32_t time_lo,
                                                    wl_fixed_t dx_w,
                                                    wl_fixed_t dy_w,
                                                    wl_fixed_t dx_unaccel_w,
                                                    wl_fixed_t dy_unaccel_w)
{
    SDL_WaylandSeat *seat = data;

    if (seat->display->relative_mode_enabled) {
        SDL_WindowData *window = seat->pointer.focus;

        // Relative motion follows keyboard focus.
        if (Wayland_SeatHasRelativePointerFocus(seat)) {
            SDL_Mouse *mouse = SDL_GetMouse();

            // Relative pointer event times are in microsecond granularity.
            const Uint64 timestamp = Wayland_AdjustEventTimestampBase(SDL_US_TO_NS(((Uint64)time_hi << 32) | (Uint64)time_lo));

            double dx;
            double dy;
            if (mouse->InputTransform || !mouse->enable_relative_system_scale) {
                dx = wl_fixed_to_double(dx_unaccel_w);
                dy = wl_fixed_to_double(dy_unaccel_w);
            } else {
                dx = wl_fixed_to_double(dx_w) * window->pointer_scale.x;
                dy = wl_fixed_to_double(dy_w) * window->pointer_scale.y;
            }

            SDL_SendMouseMotion(timestamp, window->sdlwindow, seat->pointer.sdl_id, true, (float)dx, (float)dy);
        }
    }
}

static const struct zwp_relative_pointer_v1_listener relative_pointer_listener = {
    relative_pointer_handle_relative_motion,
};

static void locked_pointer_locked(void *data,
                                  struct zwp_locked_pointer_v1 *locked_pointer)
{
}

static void locked_pointer_unlocked(void *data,
                                    struct zwp_locked_pointer_v1 *locked_pointer)
{
}

static const struct zwp_locked_pointer_v1_listener locked_pointer_listener = {
    locked_pointer_locked,
    locked_pointer_unlocked,
};

static void confined_pointer_confined(void *data,
                                      struct zwp_confined_pointer_v1 *confined_pointer)
{
}

static void confined_pointer_unconfined(void *data,
                                        struct zwp_confined_pointer_v1 *confined_pointer)
{
}

static const struct zwp_confined_pointer_v1_listener confined_pointer_listener = {
    confined_pointer_confined,
    confined_pointer_unconfined,
};

static void touch_handler_down(void *data, struct wl_touch *touch, uint32_t serial,
                               uint32_t timestamp, struct wl_surface *surface,
                               int id, wl_fixed_t fx, wl_fixed_t fy)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    SDL_WindowData *window_data;

    // Check that this surface is valid.
    if (!surface) {
        return;
    }

    Wayland_SeatAddTouch(seat, id, fx, fy, surface);
    Wayland_UpdateImplicitGrabSerial(seat, serial);
    window_data = Wayland_GetWindowDataForOwnedSurface(surface);

    if (window_data) {
        float x, y;

        if (window_data->current.logical_width <= 1) {
            x = 0.5f;
        } else {
            x = (float)wl_fixed_to_double(fx) / (window_data->current.logical_width - 1);
        }
        if (window_data->current.logical_height <= 1) {
            y = 0.5f;
        } else {
            y = (float)wl_fixed_to_double(fy) / (window_data->current.logical_height - 1);
        }

        SDL_SetMouseFocus(window_data->sdlwindow);

        SDL_SendTouch(Wayland_GetTouchTimestamp(seat, timestamp), (SDL_TouchID)(uintptr_t)touch,
                      (SDL_FingerID)(id + 1), window_data->sdlwindow, SDL_EVENT_FINGER_DOWN, x, y, 1.0f);
    }
}

static void touch_handler_up(void *data, struct wl_touch *touch, uint32_t serial,
                             uint32_t timestamp, int id)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    wl_fixed_t fx = 0, fy = 0;
    struct wl_surface *surface = NULL;

    Wayland_SeatRemoveTouch(seat, id, &fx, &fy, &surface);

    if (surface) {
        SDL_WindowData *window_data = (SDL_WindowData *)wl_surface_get_user_data(surface);

        if (window_data) {
            const float x = (float)wl_fixed_to_double(fx) / window_data->current.logical_width;
            const float y = (float)wl_fixed_to_double(fy) / window_data->current.logical_height;

            SDL_SendTouch(Wayland_GetTouchTimestamp(seat, timestamp), (SDL_TouchID)(uintptr_t)touch,
                          (SDL_FingerID)(id + 1), window_data->sdlwindow, SDL_EVENT_FINGER_UP, x, y, 0.0f);

            /* If the window currently has mouse focus, the keyboard focus is another window or NULL, the window has no
             * pointers active on it, and the surface has no active touch events, then consider mouse focus to be lost.
             */
            if (SDL_GetMouseFocus() == window_data->sdlwindow && seat->keyboard.focus != window_data &&
                !window_data->pointer_focus_count && !Wayland_SurfaceHasActiveTouches(seat->display, surface)) {
                SDL_SetMouseFocus(NULL);
            }
        }
    }
}

static void touch_handler_motion(void *data, struct wl_touch *touch, uint32_t timestamp,
                                 int id, wl_fixed_t fx, wl_fixed_t fy)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    struct wl_surface *surface = NULL;

    Wayland_SeatUpdateTouch(seat, id, fx, fy, &surface);

    if (surface) {
        SDL_WindowData *window_data = (SDL_WindowData *)wl_surface_get_user_data(surface);

        if (window_data) {
            const float x = (float)wl_fixed_to_double(fx) / window_data->current.logical_width;
            const float y = (float)wl_fixed_to_double(fy) / window_data->current.logical_height;

            SDL_SendTouchMotion(Wayland_GetTouchTimestamp(seat, timestamp), (SDL_TouchID)(uintptr_t)touch,
                                (SDL_FingerID)(id + 1), window_data->sdlwindow, x, y, 1.0f);
        }
    }
}

static void touch_handler_frame(void *data, struct wl_touch *touch)
{
}

static void touch_handler_cancel(void *data, struct wl_touch *touch)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    struct SDL_WaylandTouchPoint *tp, *temp;

    wl_list_for_each_safe (tp, temp, &seat->touch.points, link) {
        bool removed = false;

        if (tp->surface) {
            SDL_WindowData *window_data = (SDL_WindowData *)wl_surface_get_user_data(tp->surface);

            if (window_data) {
                const float x = (float)(wl_fixed_to_double(tp->fx) / window_data->current.logical_width);
                const float y = (float)(wl_fixed_to_double(tp->fy) / window_data->current.logical_height);

                SDL_SendTouch(0, (SDL_TouchID)(uintptr_t)touch,
                              (SDL_FingerID)(tp->id + 1), window_data->sdlwindow, SDL_EVENT_FINGER_CANCELED, x, y, 0.0f);

                // Remove the touch from the list before checking for still-active touches on the surface.
                WAYLAND_wl_list_remove(&tp->link);
                removed = true;

                /* If the window currently has mouse focus, the keyboard focus is another window or NULL, the window has no
                 * pointers active on it, and the surface has no active touch events, then consider mouse focus to be lost.
                 */
                if (SDL_GetMouseFocus() == window_data->sdlwindow && seat->keyboard.focus != window_data &&
                    !window_data->pointer_focus_count && !Wayland_SurfaceHasActiveTouches(seat->display, tp->surface)) {
                    SDL_SetMouseFocus(NULL);
                }
            }
        }

        if (!removed) {
            WAYLAND_wl_list_remove(&tp->link);
        }
        SDL_free(tp);
    }
}

static void touch_handler_shape(void *data, struct wl_touch *wl_touch, int32_t id, wl_fixed_t major, wl_fixed_t minor)
{
}

static void touch_handler_orientation(void *data, struct wl_touch *wl_touch, int32_t id, wl_fixed_t orientation)
{
}

static const struct wl_touch_listener touch_listener = {
    touch_handler_down,
    touch_handler_up,
    touch_handler_motion,
    touch_handler_frame,
    touch_handler_cancel,
    touch_handler_shape,      // Version 6
    touch_handler_orientation // Version 6
};

typedef struct Wayland_KeymapBuilderState
{
    SDL_Keymap *keymap;
    struct xkb_state *state;
    SDL_Keymod modstate;
} Wayland_KeymapBuilderState;

static void Wayland_keymap_iter(struct xkb_keymap *keymap, xkb_keycode_t key, void *data)
{
    Wayland_KeymapBuilderState *sdlKeymap = (Wayland_KeymapBuilderState *)data;
    const xkb_keysym_t *syms;
    const SDL_Scancode scancode = SDL_GetScancodeFromTable(SDL_SCANCODE_TABLE_XFREE86_2, (key - 8));
    if (scancode == SDL_SCANCODE_UNKNOWN) {
        return;
    }

    if (WAYLAND_xkb_state_key_get_syms(sdlKeymap->state, key, &syms) > 0) {
        SDL_Keycode keycode = SDL_GetKeyCodeFromKeySym(syms[0], key, sdlKeymap->modstate);

        if (!keycode) {
            switch (scancode) {
            case SDL_SCANCODE_RETURN:
                keycode = SDLK_RETURN;
                break;
            case SDL_SCANCODE_ESCAPE:
                keycode = SDLK_ESCAPE;
                break;
            case SDL_SCANCODE_BACKSPACE:
                keycode = SDLK_BACKSPACE;
                break;
            case SDL_SCANCODE_DELETE:
                keycode = SDLK_DELETE;
                break;
            default:
                keycode = SDL_SCANCODE_TO_KEYCODE(scancode);
                break;
            }
        }

        SDL_SetKeymapEntry(sdlKeymap->keymap, scancode, sdlKeymap->modstate, keycode);
    }
}

static void Wayland_UpdateKeymap(SDL_WaylandSeat *seat)
{
    struct Keymod_masks
    {
        SDL_Keymod sdl_mask;
        xkb_mod_mask_t xkb_mask;
    } const keymod_masks[] = {
        { SDL_KMOD_NONE, 0 },
        { SDL_KMOD_SHIFT, seat->keyboard.xkb.idx_shift },
        { SDL_KMOD_CAPS, seat->keyboard.xkb.idx_caps },
        { SDL_KMOD_SHIFT | SDL_KMOD_CAPS, seat->keyboard.xkb.idx_shift | seat->keyboard.xkb.idx_caps },
        { SDL_KMOD_MODE, seat->keyboard.xkb.idx_mod5 },
        { SDL_KMOD_MODE | SDL_KMOD_SHIFT, seat->keyboard.xkb.idx_mod5 | seat->keyboard.xkb.idx_shift },
        { SDL_KMOD_MODE | SDL_KMOD_CAPS, seat->keyboard.xkb.idx_mod5 | seat->keyboard.xkb.idx_caps },
        { SDL_KMOD_MODE | SDL_KMOD_SHIFT | SDL_KMOD_CAPS, seat->keyboard.xkb.idx_mod5 | seat->keyboard.xkb.idx_shift | seat->keyboard.xkb.idx_caps },
        { SDL_KMOD_LEVEL5, seat->keyboard.xkb.idx_mod3 },
        { SDL_KMOD_LEVEL5 | SDL_KMOD_SHIFT, seat->keyboard.xkb.idx_mod3 | seat->keyboard.xkb.idx_shift },
        { SDL_KMOD_LEVEL5 | SDL_KMOD_CAPS, seat->keyboard.xkb.idx_mod3 | seat->keyboard.xkb.idx_caps },
        { SDL_KMOD_LEVEL5 | SDL_KMOD_SHIFT | SDL_KMOD_CAPS, seat->keyboard.xkb.idx_mod3 | seat->keyboard.xkb.idx_shift | seat->keyboard.xkb.idx_caps },
        { SDL_KMOD_LEVEL5 | SDL_KMOD_MODE, seat->keyboard.xkb.idx_mod3 | seat->keyboard.xkb.idx_mod5 },
        { SDL_KMOD_LEVEL5 | SDL_KMOD_MODE | SDL_KMOD_SHIFT, seat->keyboard.xkb.idx_mod3 | seat->keyboard.xkb.idx_mod5 | seat->keyboard.xkb.idx_shift },
        { SDL_KMOD_LEVEL5 | SDL_KMOD_MODE | SDL_KMOD_CAPS, seat->keyboard.xkb.idx_mod3 | seat->keyboard.xkb.idx_mod5 | seat->keyboard.xkb.idx_caps },
        { SDL_KMOD_LEVEL5 | SDL_KMOD_MODE | SDL_KMOD_SHIFT | SDL_KMOD_CAPS, seat->keyboard.xkb.idx_mod3 | seat->keyboard.xkb.idx_mod5 | seat->keyboard.xkb.idx_shift | seat->keyboard.xkb.idx_caps },
    };

    if (!seat->keyboard.is_virtual) {
        Wayland_KeymapBuilderState keymap;

        keymap.keymap = SDL_CreateKeymap(false);
        if (!keymap.keymap) {
            return;
        }

        keymap.state = WAYLAND_xkb_state_new(seat->keyboard.xkb.keymap);
        if (!keymap.state) {
            SDL_SetError("failed to create XKB state");
            SDL_DestroyKeymap(keymap.keymap);
            return;
        }

        for (int i = 0; i < SDL_arraysize(keymod_masks); ++i) {
            keymap.modstate = keymod_masks[i].sdl_mask;
            WAYLAND_xkb_state_update_mask(keymap.state,
                                          keymod_masks[i].xkb_mask & (seat->keyboard.xkb.idx_shift | seat->keyboard.xkb.idx_mod5 | seat->keyboard.xkb.idx_mod3), 0, keymod_masks[i].xkb_mask & seat->keyboard.xkb.idx_caps,
                                          0, 0, seat->keyboard.xkb.current_group);
            WAYLAND_xkb_keymap_key_for_each(seat->keyboard.xkb.keymap,
                                            Wayland_keymap_iter,
                                            &keymap);
        }

        WAYLAND_xkb_state_unref(keymap.state);
        SDL_SetKeymap(keymap.keymap, true);
        SDL_DestroyKeymap(seat->keyboard.sdl_keymap);
        seat->keyboard.sdl_keymap = keymap.keymap;
    } else {
        // Virtual keyboards use the default keymap.
        SDL_SetKeymap(NULL, true);
        SDL_DestroyKeymap(seat->keyboard.sdl_keymap);
        seat->keyboard.sdl_keymap = NULL;
    }
}

static void keyboard_handle_keymap(void *data, struct wl_keyboard *keyboard,
                                   uint32_t format, int fd, uint32_t size)
{
    SDL_WaylandSeat *seat = data;
    char *map_str;
    const char *locale;

    if (!data) {
        close(fd);
        return;
    }

    if (format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1) {
        close(fd);
        return;
    }

    map_str = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map_str == MAP_FAILED) {
        close(fd);
        return;
    }

    if (seat->keyboard.xkb.keymap != NULL) {
        /* if there's already a keymap loaded, throw it away rather than leaking it before
         * parsing the new one
         */
        WAYLAND_xkb_keymap_unref(seat->keyboard.xkb.keymap);
        seat->keyboard.xkb.keymap = NULL;
    }
    seat->keyboard.xkb.keymap = WAYLAND_xkb_keymap_new_from_string(seat->display->xkb_context,
                                                           map_str,
                                                           XKB_KEYMAP_FORMAT_TEXT_V1,
                                                           0);
    munmap(map_str, size);
    close(fd);

    if (!seat->keyboard.xkb.keymap) {
        SDL_SetError("failed to compile keymap");
        return;
    }

#define GET_MOD_INDEX(mod) \
    WAYLAND_xkb_keymap_mod_get_index(seat->keyboard.xkb.keymap, XKB_MOD_NAME_##mod)
    seat->keyboard.xkb.idx_shift = 1 << GET_MOD_INDEX(SHIFT);
    seat->keyboard.xkb.idx_ctrl = 1 << GET_MOD_INDEX(CTRL);
    seat->keyboard.xkb.idx_alt = 1 << GET_MOD_INDEX(ALT);
    seat->keyboard.xkb.idx_gui = 1 << GET_MOD_INDEX(LOGO);
    seat->keyboard.xkb.idx_mod3 = 1 << GET_MOD_INDEX(MOD3);
    seat->keyboard.xkb.idx_mod5 = 1 << GET_MOD_INDEX(MOD5);
    seat->keyboard.xkb.idx_num = 1 << GET_MOD_INDEX(NUM);
    seat->keyboard.xkb.idx_caps = 1 << GET_MOD_INDEX(CAPS);
#undef GET_MOD_INDEX

    if (seat->keyboard.xkb.state != NULL) {
        /* if there's already a state, throw it away rather than leaking it before
         * trying to create a new one with the new keymap.
         */
        WAYLAND_xkb_state_unref(seat->keyboard.xkb.state);
        seat->keyboard.xkb.state = NULL;
    }
    seat->keyboard.xkb.state = WAYLAND_xkb_state_new(seat->keyboard.xkb.keymap);
    if (!seat->keyboard.xkb.state) {
        SDL_SetError("failed to create XKB state");
        WAYLAND_xkb_keymap_unref(seat->keyboard.xkb.keymap);
        seat->keyboard.xkb.keymap = NULL;
        return;
    }

    /*
     * Assume that a nameless layout implies a virtual keyboard with an arbitrary layout.
     * TODO: Use a better method of detection?
     */
    seat->keyboard.is_virtual = WAYLAND_xkb_keymap_layout_get_name(seat->keyboard.xkb.keymap, 0) == NULL;

    // Update the keymap if changed.
    if (seat->keyboard.xkb.current_group != XKB_GROUP_INVALID) {
        Wayland_UpdateKeymap(seat);
    }

    /*
     * See https://blogs.s-osg.org/compose-key-support-weston/
     * for further explanation on dead keys in Wayland.
     */

    // Look up the preferred locale, falling back to "C" as default
    locale = SDL_getenv("LC_ALL");
    if (!locale) {
        locale = SDL_getenv("LC_CTYPE");
        if (!locale) {
            locale = SDL_getenv("LANG");
            if (!locale) {
                locale = "C";
            }
        }
    }

    // Set up XKB compose table
    if (seat->keyboard.xkb.compose_table != NULL) {
        WAYLAND_xkb_compose_table_unref(seat->keyboard.xkb.compose_table);
        seat->keyboard.xkb.compose_table = NULL;
    }
    seat->keyboard.xkb.compose_table = WAYLAND_xkb_compose_table_new_from_locale(seat->display->xkb_context,
                                                                         locale, XKB_COMPOSE_COMPILE_NO_FLAGS);
    if (seat->keyboard.xkb.compose_table) {
        // Set up XKB compose state
        if (seat->keyboard.xkb.compose_state != NULL) {
            WAYLAND_xkb_compose_state_unref(seat->keyboard.xkb.compose_state);
            seat->keyboard.xkb.compose_state = NULL;
        }
        seat->keyboard.xkb.compose_state = WAYLAND_xkb_compose_state_new(seat->keyboard.xkb.compose_table,
                                                                 XKB_COMPOSE_STATE_NO_FLAGS);
        if (!seat->keyboard.xkb.compose_state) {
            SDL_SetError("could not create XKB compose state");
            WAYLAND_xkb_compose_table_unref(seat->keyboard.xkb.compose_table);
            seat->keyboard.xkb.compose_table = NULL;
        }
    }
}

/*
 * Virtual keyboards can have arbitrary layouts, arbitrary scancodes/keycodes, etc...
 * Key presses from these devices must be looked up by their keysym value.
 */
static SDL_Scancode Wayland_GetScancodeForKey(SDL_WaylandSeat *seat, uint32_t key)
{
    SDL_Scancode scancode = SDL_SCANCODE_UNKNOWN;

    if (!seat->keyboard.is_virtual) {
        scancode = SDL_GetScancodeFromTable(SDL_SCANCODE_TABLE_XFREE86_2, key);
    } else {
        const xkb_keysym_t *syms;
        if (WAYLAND_xkb_keymap_key_get_syms_by_level(seat->keyboard.xkb.keymap, key + 8, seat->keyboard.xkb.current_group, 0, &syms) > 0) {
            scancode = SDL_GetScancodeFromKeySym(syms[0], key);
        }
    }

    return scancode;
}

static void Wayland_ReconcileModifiers(SDL_WaylandSeat *seat, bool key_pressed)
{
    /* Handle explicit pressed modifier state. This will correct the modifier state
     * if common modifier keys were remapped and the modifiers presumed to be set
     * during a key press event were incorrect, or if the modifier was set to the
     * pressed state via means other than pressing the physical key.
     */
    if (!key_pressed) {
        if (seat->keyboard.xkb.wl_pressed_modifiers & seat->keyboard.xkb.idx_shift) {
            if (!(seat->keyboard.pressed_modifiers & SDL_KMOD_SHIFT)) {
                seat->keyboard.pressed_modifiers |= SDL_KMOD_SHIFT;
            }
        } else {
            seat->keyboard.pressed_modifiers &= ~SDL_KMOD_SHIFT;
        }

        if (seat->keyboard.xkb.wl_pressed_modifiers & seat->keyboard.xkb.idx_ctrl) {
            if (!(seat->keyboard.pressed_modifiers & SDL_KMOD_CTRL)) {
                seat->keyboard.pressed_modifiers |= SDL_KMOD_CTRL;
            }
        } else {
            seat->keyboard.pressed_modifiers &= ~SDL_KMOD_CTRL;
        }

        if (seat->keyboard.xkb.wl_pressed_modifiers & seat->keyboard.xkb.idx_alt) {
            if (!(seat->keyboard.pressed_modifiers & SDL_KMOD_ALT)) {
                seat->keyboard.pressed_modifiers |= SDL_KMOD_ALT;
            }
        } else {
            seat->keyboard.pressed_modifiers &= ~SDL_KMOD_ALT;
        }

        if (seat->keyboard.xkb.wl_pressed_modifiers & seat->keyboard.xkb.idx_gui) {
            if (!(seat->keyboard.pressed_modifiers & SDL_KMOD_GUI)) {
                seat->keyboard.pressed_modifiers |= SDL_KMOD_GUI;
            }
        } else {
            seat->keyboard.pressed_modifiers &= ~SDL_KMOD_GUI;
        }

        /* Note: This is not backwards: in the default keymap, Mod5 is typically
         *       level 3 shift, and Mod3 is typically level 5 shift.
         */
        if (seat->keyboard.xkb.wl_pressed_modifiers & seat->keyboard.xkb.idx_mod3) {
            if (!(seat->keyboard.pressed_modifiers & SDL_KMOD_LEVEL5)) {
                seat->keyboard.pressed_modifiers |= SDL_KMOD_LEVEL5;
            }
        } else {
            seat->keyboard.pressed_modifiers &= ~SDL_KMOD_LEVEL5;
        }

        if (seat->keyboard.xkb.wl_pressed_modifiers & seat->keyboard.xkb.idx_mod5) {
            if (!(seat->keyboard.pressed_modifiers & SDL_KMOD_MODE)) {
                seat->keyboard.pressed_modifiers |= SDL_KMOD_MODE;
            }
        } else {
            seat->keyboard.pressed_modifiers &= ~SDL_KMOD_MODE;
        }
    }

    /* If a latch or lock was activated by a keypress, the latch/lock will
     * be tied to the specific left/right key that initiated it. Otherwise,
     * the ambiguous left/right combo is used.
     *
     * The modifier will remain active until the latch/lock is released by
     * the system.
     */
    if (seat->keyboard.xkb.wl_locked_modifiers & seat->keyboard.xkb.idx_shift) {
        if (seat->keyboard.pressed_modifiers & SDL_KMOD_SHIFT) {
            seat->keyboard.locked_modifiers &= ~SDL_KMOD_SHIFT;
            seat->keyboard.locked_modifiers |= (seat->keyboard.pressed_modifiers & SDL_KMOD_SHIFT);
        } else if (!(seat->keyboard.locked_modifiers & SDL_KMOD_SHIFT)) {
            seat->keyboard.locked_modifiers |= SDL_KMOD_SHIFT;
        }
    } else {
        seat->keyboard.locked_modifiers &= ~SDL_KMOD_SHIFT;
    }

    if (seat->keyboard.xkb.wl_locked_modifiers & seat->keyboard.xkb.idx_ctrl) {
        if (seat->keyboard.pressed_modifiers & SDL_KMOD_CTRL) {
            seat->keyboard.locked_modifiers &= ~SDL_KMOD_CTRL;
            seat->keyboard.locked_modifiers |= (seat->keyboard.pressed_modifiers & SDL_KMOD_CTRL);
        } else if (!(seat->keyboard.locked_modifiers & SDL_KMOD_CTRL)) {
            seat->keyboard.locked_modifiers |= SDL_KMOD_CTRL;
        }
    } else {
        seat->keyboard.locked_modifiers &= ~SDL_KMOD_CTRL;
    }

    if (seat->keyboard.xkb.wl_locked_modifiers & seat->keyboard.xkb.idx_alt) {
        if (seat->keyboard.pressed_modifiers & SDL_KMOD_ALT) {
            seat->keyboard.locked_modifiers &= ~SDL_KMOD_ALT;
            seat->keyboard.locked_modifiers |= (seat->keyboard.pressed_modifiers & SDL_KMOD_ALT);
        } else if (!(seat->keyboard.locked_modifiers & SDL_KMOD_ALT)) {
            seat->keyboard.locked_modifiers |= SDL_KMOD_ALT;
        }
    } else {
        seat->keyboard.locked_modifiers &= ~SDL_KMOD_ALT;
    }

    if (seat->keyboard.xkb.wl_locked_modifiers & seat->keyboard.xkb.idx_gui) {
        if (seat->keyboard.pressed_modifiers & SDL_KMOD_GUI) {
            seat->keyboard.locked_modifiers &= ~SDL_KMOD_GUI;
            seat->keyboard.locked_modifiers |= (seat->keyboard.pressed_modifiers & SDL_KMOD_GUI);
        } else if (!(seat->keyboard.locked_modifiers & SDL_KMOD_GUI)) {
            seat->keyboard.locked_modifiers |= SDL_KMOD_GUI;
        }
    } else {
        seat->keyboard.locked_modifiers &= ~SDL_KMOD_GUI;
    }

    // As above, this is correct: Mod3 is typically level 5 shift, and Mod5 is typically level 3 shift.
    if (seat->keyboard.xkb.wl_locked_modifiers & seat->keyboard.xkb.idx_mod3) {
        seat->keyboard.locked_modifiers |= SDL_KMOD_LEVEL5;
    } else {
        seat->keyboard.locked_modifiers &= ~SDL_KMOD_LEVEL5;
    }

    if (seat->keyboard.xkb.wl_locked_modifiers & seat->keyboard.xkb.idx_mod5) {
        seat->keyboard.locked_modifiers |= SDL_KMOD_MODE;
    } else {
        seat->keyboard.locked_modifiers &= ~SDL_KMOD_MODE;
    }

    // Capslock and Numlock can only be locked, not pressed.
    if (seat->keyboard.xkb.wl_locked_modifiers & seat->keyboard.xkb.idx_caps) {
        seat->keyboard.locked_modifiers |= SDL_KMOD_CAPS;
    } else {
        seat->keyboard.locked_modifiers &= ~SDL_KMOD_CAPS;
    }

    if (seat->keyboard.xkb.wl_locked_modifiers & seat->keyboard.xkb.idx_num) {
        seat->keyboard.locked_modifiers |= SDL_KMOD_NUM;
    } else {
        seat->keyboard.locked_modifiers &= ~SDL_KMOD_NUM;
    }

    SDL_SetModState(seat->keyboard.pressed_modifiers | seat->keyboard.locked_modifiers);
}

static void Wayland_HandleModifierKeys(SDL_WaylandSeat *seat, SDL_Scancode scancode, bool pressed)
{
    const SDL_Keycode keycode = SDL_GetKeyFromScancode(scancode, SDL_KMOD_NONE, false);
    SDL_Keymod mod;

    /* SDL clients expect modifier state to be activated at the same time as the
     * source keypress, so we set pressed modifier state with the usual modifier
     * keys here, as the explicit modifier event won't arrive until after the
     * keypress event. If this is wrong, it will be corrected when the explicit
     * modifier state is sent at a later time.
     */
    switch (keycode) {
    case SDLK_LSHIFT:
        mod = SDL_KMOD_LSHIFT;
        break;
    case SDLK_RSHIFT:
        mod = SDL_KMOD_RSHIFT;
        break;
    case SDLK_LCTRL:
        mod = SDL_KMOD_LCTRL;
        break;
    case SDLK_RCTRL:
        mod = SDL_KMOD_RCTRL;
        break;
    case SDLK_LALT:
        mod = SDL_KMOD_LALT;
        break;
    case SDLK_RALT:
        mod = SDL_KMOD_RALT;
        break;
    case SDLK_LGUI:
        mod = SDL_KMOD_LGUI;
        break;
    case SDLK_RGUI:
        mod = SDL_KMOD_RGUI;
        break;
    case SDLK_MODE:
        mod = SDL_KMOD_MODE;
        break;
    case SDLK_LEVEL5_SHIFT:
        mod = SDL_KMOD_LEVEL5;
        break;
    default:
        return;
    }

    if (pressed) {
        seat->keyboard.pressed_modifiers |= mod;
    } else {
        seat->keyboard.pressed_modifiers &= ~mod;
    }

    Wayland_ReconcileModifiers(seat, true);
}

static void keyboard_handle_enter(void *data, struct wl_keyboard *keyboard,
                                  uint32_t serial, struct wl_surface *surface,
                                  struct wl_array *keys)
{
    SDL_WaylandSeat *seat = data;
    uint32_t *key;

    if (!surface) {
        // Enter event for a destroyed surface.
        return;
    }

    SDL_WindowData *window = Wayland_GetWindowDataForOwnedSurface(surface);
    if (!window) {
        // Not a surface owned by SDL.
        return;
    }

    ++window->keyboard_focus_count;
    seat->keyboard.focus = window;

    // Restore the keyboard focus to the child popup that was holding it
    SDL_SetKeyboardFocus(window->sdlwindow->keyboard_focus ? window->sdlwindow->keyboard_focus : window->sdlwindow);

    // Update the keyboard grab and any relative pointer grabs related to this keyboard focus.
    Wayland_SeatUpdateKeyboardGrab(seat);
    Wayland_DisplayUpdatePointerGrabs(seat->display, window);

    // Update text input and IME focus.
    Wayland_UpdateTextInput(seat->display);

#ifdef SDL_USE_IME
    if (!seat->text_input.zwp_text_input) {
        SDL_IME_SetFocus(true);
    }
#endif

    Uint64 timestamp = SDL_GetTicksNS();
    window->last_focus_event_time_ns = timestamp;

    if (SDL_GetCurrentKeymap() != seat->keyboard.sdl_keymap) {
        SDL_SetKeymap(seat->keyboard.sdl_keymap, true);
    }

    wl_array_for_each (key, keys) {
        const SDL_Scancode scancode = Wayland_GetScancodeForKey(seat, *key);
        const SDL_Keycode keycode = SDL_GetKeyFromScancode(scancode, SDL_KMOD_NONE, false);

        switch (keycode) {
        case SDLK_LSHIFT:
        case SDLK_RSHIFT:
        case SDLK_LCTRL:
        case SDLK_RCTRL:
        case SDLK_LALT:
        case SDLK_RALT:
        case SDLK_LGUI:
        case SDLK_RGUI:
        case SDLK_MODE:
        case SDLK_LEVEL5_SHIFT:
            Wayland_HandleModifierKeys(seat, scancode, true);
            SDL_SendKeyboardKeyIgnoreModifiers(timestamp, seat->keyboard.sdl_id, *key, scancode, true);
            break;
        default:
            break;
        }
    }
}

static void keyboard_handle_leave(void *data, struct wl_keyboard *keyboard,
                                  uint32_t serial, struct wl_surface *surface)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;

    if (!surface) {
        // Leave event for a destroyed surface.
        return;
    }

    SDL_WindowData *window = Wayland_GetWindowDataForOwnedSurface(surface);
    if (!window) {
        // Not a surface owned by SDL.
        return;
    }

    // Stop key repeat before clearing keyboard focus
    keyboard_repeat_clear(&seat->keyboard.repeat);

    SDL_Window *keyboard_focus = SDL_GetKeyboardFocus();

    // The keyboard focus may be a child popup
    while (keyboard_focus && SDL_WINDOW_IS_POPUP(keyboard_focus)) {
        keyboard_focus = keyboard_focus->parent;
    }

    const bool had_focus = keyboard_focus && window->sdlwindow == keyboard_focus;
    seat->keyboard.focus = NULL;
    --window->keyboard_focus_count;

    // Only relinquish focus if this window has the active focus, and no other keyboards have focus on the window.
    if (!window->keyboard_focus_count && had_focus) {
        SDL_SetKeyboardFocus(NULL);
    }

    // Release the keyboard grab and any relative pointer grabs related to this keyboard focus.
    Wayland_SeatUpdateKeyboardGrab(seat);
    Wayland_DisplayUpdatePointerGrabs(seat->display, window);

    // Clear the pressed modifiers.
    seat->keyboard.pressed_modifiers = SDL_KMOD_NONE;

    // Update text input and IME focus.
    Wayland_UpdateTextInput(seat->display);

#ifdef SDL_USE_IME
    if (!seat->text_input.zwp_text_input && !window->keyboard_focus_count) {
        SDL_IME_SetFocus(false);
    }
#endif

    /* If the window has mouse focus, has no pointers within it, and no active touches, consider
     * mouse focus to be lost.
     */
    if (SDL_GetMouseFocus() == window->sdlwindow && !window->pointer_focus_count &&
        !Wayland_SurfaceHasActiveTouches(seat->display, surface)) {
        SDL_SetMouseFocus(NULL);
    }
}

static bool keyboard_input_get_text(char text[8], const SDL_WaylandSeat *seat, uint32_t key, bool down, bool *handled_by_ime)
{
    const xkb_keysym_t *syms;
    xkb_keysym_t sym;

    if (!seat->keyboard.focus || !seat->keyboard.xkb.state) {
        return false;
    }

    // TODO: Can this happen?
    if (WAYLAND_xkb_state_key_get_syms(seat->keyboard.xkb.state, key + 8, &syms) != 1) {
        return false;
    }
    sym = syms[0];

#ifdef SDL_USE_IME
    if (SDL_IME_ProcessKeyEvent(sym, key + 8, down)) {
        if (handled_by_ime) {
            *handled_by_ime = true;
        }
        return true;
    }
#endif

    if (!down) {
        return false;
    }

    if (seat->keyboard.xkb.compose_state && WAYLAND_xkb_compose_state_feed(seat->keyboard.xkb.compose_state, sym) == XKB_COMPOSE_FEED_ACCEPTED) {
        switch (WAYLAND_xkb_compose_state_get_status(seat->keyboard.xkb.compose_state)) {
        case XKB_COMPOSE_COMPOSING:
            if (handled_by_ime) {
                *handled_by_ime = true;
            }
            return true;
        case XKB_COMPOSE_CANCELLED:
        default:
            sym = XKB_KEY_NoSymbol;
            break;
        case XKB_COMPOSE_NOTHING:
            break;
        case XKB_COMPOSE_COMPOSED:
            sym = WAYLAND_xkb_compose_state_get_one_sym(seat->keyboard.xkb.compose_state);
            break;
        }
    }

    return WAYLAND_xkb_keysym_to_utf8(sym, text, 8) > 0;
}

static void keyboard_handle_key(void *data, struct wl_keyboard *keyboard,
                                uint32_t serial, uint32_t time, uint32_t key,
                                uint32_t state_w)
{
    SDL_WaylandSeat *seat = data;
    enum wl_keyboard_key_state state = state_w;
    char text[8];
    bool has_text = false;
    bool handled_by_ime = false;
    const Uint64 timestamp_raw_ns = Wayland_GetKeyboardTimestampRaw(seat, time);

    Wayland_UpdateImplicitGrabSerial(seat, serial);

    if (seat->keyboard.sdl_keymap != SDL_GetCurrentKeymap()) {
        SDL_SetKeymap(seat->keyboard.sdl_keymap, true);
        SDL_SetModState(seat->keyboard.pressed_modifiers | seat->keyboard.locked_modifiers);
    }

    if (state == WL_KEYBOARD_KEY_STATE_PRESSED) {
        SDL_Window *keyboard_focus = SDL_GetKeyboardFocus();
        if (keyboard_focus && SDL_TextInputActive(keyboard_focus)) {
            has_text = keyboard_input_get_text(text, seat, key, true, &handled_by_ime);
        }
    } else {
        if (keyboard_repeat_key_is_set(&seat->keyboard.repeat, key)) {
            /* Send any due key repeat events before stopping the repeat and generating the key up event.
             * Compute time based on the Wayland time, as it reports when the release event happened.
             * Using SDL_GetTicks would be wrong, as it would report when the release event is processed,
             * which may be off if the application hasn't pumped events for a while.
             */
            keyboard_repeat_handle(&seat->keyboard.repeat, timestamp_raw_ns - seat->keyboard.repeat.wl_press_time_ns);
            keyboard_repeat_clear(&seat->keyboard.repeat);
        }
        keyboard_input_get_text(text, seat, key, false, &handled_by_ime);
    }

    const SDL_Scancode scancode = Wayland_GetScancodeForKey(seat, key);
    Wayland_HandleModifierKeys(seat, scancode, state == WL_KEYBOARD_KEY_STATE_PRESSED);
    Uint64 timestamp = Wayland_AdjustEventTimestampBase(timestamp_raw_ns);

    SDL_SendKeyboardKeyIgnoreModifiers(timestamp, seat->keyboard.sdl_id, key, scancode, state == WL_KEYBOARD_KEY_STATE_PRESSED);

    if (state == WL_KEYBOARD_KEY_STATE_PRESSED) {
        if (has_text && !(SDL_GetModState() & (SDL_KMOD_CTRL | SDL_KMOD_ALT))) {
            if (!handled_by_ime) {
                SDL_SendKeyboardText(text);
            }
        }
        if (seat->keyboard.xkb.keymap && WAYLAND_xkb_keymap_key_repeats(seat->keyboard.xkb.keymap, key + 8)) {
            keyboard_repeat_set(&seat->keyboard.repeat, seat->keyboard.sdl_id, key, timestamp_raw_ns, scancode, has_text, text);
        }
    }
}

static void keyboard_handle_modifiers(void *data, struct wl_keyboard *keyboard,
                                      uint32_t serial, uint32_t mods_depressed,
                                      uint32_t mods_latched, uint32_t mods_locked,
                                      uint32_t group)
{
    SDL_WaylandSeat *seat = data;

    if (seat->keyboard.xkb.state == NULL) {
        /* if we get a modifier notification before the keymap, there's nothing we can do with the information
        */
        return;
    }

    WAYLAND_xkb_state_update_mask(seat->keyboard.xkb.state, mods_depressed, mods_latched,
                                  mods_locked, 0, 0, group);

    seat->keyboard.xkb.wl_pressed_modifiers = mods_depressed;
    seat->keyboard.xkb.wl_locked_modifiers = mods_latched | mods_locked;

    Wayland_ReconcileModifiers(seat, false);

    // If a key is repeating, update the text to apply the modifier.
    if (keyboard_repeat_is_set(&seat->keyboard.repeat)) {
        char text[8];
        const uint32_t key = keyboard_repeat_get_key(&seat->keyboard.repeat);

        if (keyboard_input_get_text(text, seat, key, true, NULL)) {
            keyboard_repeat_set_text(&seat->keyboard.repeat, text);
        }
    }

    if (group == seat->keyboard.xkb.current_group) {
        return;
    }

    // The layout changed, remap and fire an event. Virtual keyboards use the default keymap.
    seat->keyboard.xkb.current_group = group;
    Wayland_UpdateKeymap(seat);
}

static void keyboard_handle_repeat_info(void *data, struct wl_keyboard *wl_keyboard,
                                        int32_t rate, int32_t delay)
{
    SDL_WaylandSeat *seat = data;
    seat->keyboard.repeat.repeat_rate = SDL_clamp(rate, 0, 1000);
    seat->keyboard.repeat.repeat_delay_ms = delay;
    seat->keyboard.repeat.is_initialized = true;
}

static const struct wl_keyboard_listener keyboard_listener = {
    keyboard_handle_keymap,
    keyboard_handle_enter,
    keyboard_handle_leave,
    keyboard_handle_key,
    keyboard_handle_modifiers,
    keyboard_handle_repeat_info, // Version 4
};

static void Wayland_SeatCreateRelativePointer(SDL_WaylandSeat *seat)
{
    if (seat->display->relative_pointer_manager) {
        if (seat->pointer.wl_pointer && !seat->pointer.relative_pointer) {
            seat->pointer.relative_pointer = zwp_relative_pointer_manager_v1_get_relative_pointer(seat->display->relative_pointer_manager, seat->pointer.wl_pointer);
            zwp_relative_pointer_v1_add_listener(seat->pointer.relative_pointer,
                                                 &relative_pointer_listener,
                                                 seat);
        }
    }
}

void Wayland_DisplayInitRelativePointerManager(SDL_VideoData *display)
{
    SDL_WaylandSeat *seat;
    wl_list_for_each(seat, &display->seat_list, link) {
        Wayland_SeatCreateRelativePointer(seat);
    }
}

static void Wayland_SeatDestroyPointer(SDL_WaylandSeat *seat, bool send_event)
{
    // Make sure focus is removed from a surface before the pointer is destroyed.
    if (seat->pointer.focus) {
        pointer_handle_leave(seat, seat->pointer.wl_pointer, 0, seat->pointer.focus->surface);
    }

    SDL_RemoveMouse(seat->pointer.sdl_id, send_event);

    if (seat->pointer.confined_pointer) {
        zwp_confined_pointer_v1_destroy(seat->pointer.confined_pointer);
    }

    if (seat->pointer.locked_pointer) {
        zwp_locked_pointer_v1_destroy(seat->pointer.locked_pointer);
    }

    if (seat->pointer.relative_pointer) {
        zwp_relative_pointer_v1_destroy(seat->pointer.relative_pointer);
    }

    if (seat->pointer.timestamps) {
        zwp_input_timestamps_v1_destroy(seat->pointer.timestamps);
    }

    if (seat->pointer.cursor_state.frame_callback) {
        wl_callback_destroy(seat->pointer.cursor_state.frame_callback);
    }

    if (seat->pointer.cursor_state.surface) {
        wl_surface_destroy(seat->pointer.cursor_state.surface);
    }

    if (seat->pointer.cursor_state.viewport) {
        wp_viewport_destroy(seat->pointer.cursor_state.viewport);
    }

    if (seat->pointer.cursor_shape) {
        wp_cursor_shape_device_v1_destroy(seat->pointer.cursor_shape);
    }

    if (seat->pointer.wl_pointer) {
        if (wl_pointer_get_version(seat->pointer.wl_pointer) >= WL_POINTER_RELEASE_SINCE_VERSION) {
            wl_pointer_release(seat->pointer.wl_pointer);
        } else {
            wl_pointer_destroy(seat->pointer.wl_pointer);
        }
    }

    SDL_zero(seat->pointer);
}

static void Wayland_SeatDestroyKeyboard(SDL_WaylandSeat *seat, bool send_event)
{
    // Make sure focus is removed from a surface before the keyboard is destroyed.
    if (seat->keyboard.focus) {
        keyboard_handle_leave(seat, seat->keyboard.wl_keyboard, 0, seat->keyboard.focus->surface);
    }

    SDL_RemoveKeyboard(seat->keyboard.sdl_id, send_event);

    if (seat->keyboard.sdl_keymap) {
        if (seat->keyboard.sdl_keymap == SDL_GetCurrentKeymap()) {
            SDL_SetKeymap(NULL, false);
        }
        SDL_DestroyKeymap(seat->keyboard.sdl_keymap);
    }

    if (seat->keyboard.key_inhibitor) {
        zwp_keyboard_shortcuts_inhibitor_v1_destroy(seat->keyboard.key_inhibitor);
    }

    if (seat->keyboard.timestamps) {
        zwp_input_timestamps_v1_destroy(seat->keyboard.timestamps);
    }

    if (seat->keyboard.wl_keyboard) {
        if (wl_keyboard_get_version(seat->keyboard.wl_keyboard) >= WL_KEYBOARD_RELEASE_SINCE_VERSION) {
            wl_keyboard_release(seat->keyboard.wl_keyboard);
        } else {
            wl_keyboard_destroy(seat->keyboard.wl_keyboard);
        }
    }

    if (seat->keyboard.xkb.compose_state) {
        WAYLAND_xkb_compose_state_unref(seat->keyboard.xkb.compose_state);
    }

    if (seat->keyboard.xkb.compose_table) {
        WAYLAND_xkb_compose_table_unref(seat->keyboard.xkb.compose_table);
    }

    if (seat->keyboard.xkb.state) {
        WAYLAND_xkb_state_unref(seat->keyboard.xkb.state);
    }

    if (seat->keyboard.xkb.keymap) {
        WAYLAND_xkb_keymap_unref(seat->keyboard.xkb.keymap);
    }

    SDL_zero(seat->keyboard);
}

static void Wayland_SeatDestroyTouch(SDL_WaylandSeat *seat)
{
    // Cancel any active touches before the touch object is destroyed.
    if (seat->touch.wl_touch) {
        touch_handler_cancel(seat, seat->touch.wl_touch);
    }

    SDL_DelTouch((SDL_TouchID)(uintptr_t)seat->touch.wl_touch);

    if (seat->touch.timestamps) {
        zwp_input_timestamps_v1_destroy(seat->touch.timestamps);
    }

    if (seat->touch.wl_touch) {
        if (wl_touch_get_version(seat->touch.wl_touch) >= WL_TOUCH_RELEASE_SINCE_VERSION) {
            wl_touch_release(seat->touch.wl_touch);
        } else {
            wl_touch_destroy(seat->touch.wl_touch);
        }
    }

    SDL_zero(seat->touch);
    WAYLAND_wl_list_init(&seat->touch.points);
}

static void seat_handle_capabilities(void *data, struct wl_seat *wl_seat, enum wl_seat_capability capabilities)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    char name_fmt[256];

    if ((capabilities & WL_SEAT_CAPABILITY_POINTER) && !seat->pointer.wl_pointer) {
        seat->pointer.wl_pointer = wl_seat_get_pointer(wl_seat);
        SDL_memset(&seat->pointer.current_axis_info, 0, sizeof(seat->pointer.current_axis_info));

        Wayland_SeatCreateCursorShape(seat);

        wl_pointer_set_user_data(seat->pointer.wl_pointer, seat);
        wl_pointer_add_listener(seat->pointer.wl_pointer, &pointer_listener, seat);

        Wayland_SeatCreateRelativePointer(seat);

        seat->pointer.sdl_id = SDL_GetNextObjectID();

        if (seat->name) {
            SDL_snprintf(name_fmt, sizeof(name_fmt), "%s (%s)", WAYLAND_DEFAULT_POINTER_NAME, seat->name);
        } else {
            SDL_snprintf(name_fmt, sizeof(name_fmt), "%s %" SDL_PRIu32, WAYLAND_DEFAULT_POINTER_NAME, seat->pointer.sdl_id);
        }

        SDL_AddMouse(seat->pointer.sdl_id, name_fmt, !seat->display->initializing);
    } else if (!(capabilities & WL_SEAT_CAPABILITY_POINTER) && seat->pointer.wl_pointer) {
        Wayland_SeatDestroyPointer(seat, true);
    }

    if ((capabilities & WL_SEAT_CAPABILITY_TOUCH) && !seat->touch.wl_touch) {
        seat->touch.wl_touch = wl_seat_get_touch(wl_seat);
        wl_touch_set_user_data(seat->touch.wl_touch, seat);
        wl_touch_add_listener(seat->touch.wl_touch, &touch_listener, seat);

        if (seat->name) {
            SDL_snprintf(name_fmt, sizeof(name_fmt), "%s (%s)", WAYLAND_DEFAULT_TOUCH_NAME, seat->name);
        } else {
            SDL_snprintf(name_fmt, sizeof(name_fmt), "%s %" SDL_PRIu64, WAYLAND_DEFAULT_TOUCH_NAME, (SDL_TouchID)(uintptr_t)seat->touch.wl_touch);
        }

        SDL_AddTouch((SDL_TouchID)(uintptr_t)seat->touch.wl_touch, SDL_TOUCH_DEVICE_DIRECT, name_fmt);
    } else if (!(capabilities & WL_SEAT_CAPABILITY_TOUCH) && seat->touch.wl_touch) {
        Wayland_SeatDestroyTouch(seat);
    }

    if ((capabilities & WL_SEAT_CAPABILITY_KEYBOARD) && !seat->keyboard.wl_keyboard) {
        seat->keyboard.wl_keyboard = wl_seat_get_keyboard(wl_seat);
        wl_keyboard_set_user_data(seat->keyboard.wl_keyboard, seat);
        wl_keyboard_add_listener(seat->keyboard.wl_keyboard, &keyboard_listener, seat);

        seat->keyboard.sdl_id = SDL_GetNextObjectID();

        if (seat->name) {
            SDL_snprintf(name_fmt, sizeof(name_fmt), "%s (%s)", WAYLAND_DEFAULT_KEYBOARD_NAME, seat->name);
        } else {
            SDL_snprintf(name_fmt, sizeof(name_fmt), "%s %" SDL_PRIu32, WAYLAND_DEFAULT_KEYBOARD_NAME, seat->keyboard.sdl_id);
        }

        SDL_AddKeyboard(seat->keyboard.sdl_id, name_fmt, !seat->display->initializing);
    } else if (!(capabilities & WL_SEAT_CAPABILITY_KEYBOARD) && seat->keyboard.wl_keyboard) {
        Wayland_SeatDestroyKeyboard(seat, true);
    }

    Wayland_SeatRegisterInputTimestampListeners(seat);
}

static void seat_handle_name(void *data, struct wl_seat *wl_seat, const char *name)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    char name_fmt[256];

    if (name && *name != '\0') {
        seat->name = SDL_strdup(name);

        if (seat->keyboard.wl_keyboard) {
            SDL_snprintf(name_fmt, sizeof(name_fmt), "%s (%s)", WAYLAND_DEFAULT_KEYBOARD_NAME, seat->name);
            SDL_SetKeyboardName(seat->keyboard.sdl_id, name_fmt);
        }

        if (seat->pointer.wl_pointer) {
            SDL_snprintf(name_fmt, sizeof(name_fmt), "%s (%s)", WAYLAND_DEFAULT_POINTER_NAME, seat->name);
            SDL_SetMouseName(seat->pointer.sdl_id, name_fmt);
        }

        if (seat->touch.wl_touch) {
            SDL_snprintf(name_fmt, sizeof(name_fmt), "%s (%s)", WAYLAND_DEFAULT_TOUCH_NAME, seat->name);
            SDL_SetTouchName((SDL_TouchID)(uintptr_t)seat->touch.wl_touch, name_fmt);
        }
    }
}

static const struct wl_seat_listener seat_listener = {
    seat_handle_capabilities,
    seat_handle_name, // Version 2
};

static void data_source_handle_target(void *data, struct wl_data_source *wl_data_source,
                                      const char *mime_type)
{
}

static void data_source_handle_send(void *data, struct wl_data_source *wl_data_source,
                                    const char *mime_type, int32_t fd)
{
    Wayland_data_source_send((SDL_WaylandDataSource *)data, mime_type, fd);
}

static void data_source_handle_cancelled(void *data, struct wl_data_source *wl_data_source)
{
    SDL_WaylandDataSource *source = data;
    if (source) {
        Wayland_data_source_destroy(source);
    }
}

static void data_source_handle_dnd_drop_performed(void *data, struct wl_data_source *wl_data_source)
{
}

static void data_source_handle_dnd_finished(void *data, struct wl_data_source *wl_data_source)
{
}

static void data_source_handle_action(void *data, struct wl_data_source *wl_data_source,
                                      uint32_t dnd_action)
{
}

static const struct wl_data_source_listener data_source_listener = {
    data_source_handle_target,
    data_source_handle_send,
    data_source_handle_cancelled,
    data_source_handle_dnd_drop_performed, // Version 3
    data_source_handle_dnd_finished,       // Version 3
    data_source_handle_action,             // Version 3
};

static void primary_selection_source_send(void *data, struct zwp_primary_selection_source_v1 *zwp_primary_selection_source_v1,
                                          const char *mime_type, int32_t fd)
{
    Wayland_primary_selection_source_send((SDL_WaylandPrimarySelectionSource *)data,
                                          mime_type, fd);
}

static void primary_selection_source_cancelled(void *data, struct zwp_primary_selection_source_v1 *zwp_primary_selection_source_v1)
{
    Wayland_primary_selection_source_destroy(data);
}

static const struct zwp_primary_selection_source_v1_listener primary_selection_source_listener = {
    primary_selection_source_send,
    primary_selection_source_cancelled,
};

SDL_WaylandDataSource *Wayland_data_source_create(SDL_VideoDevice *_this)
{
    SDL_WaylandDataSource *data_source = NULL;
    SDL_VideoData *driver_data = NULL;
    struct wl_data_source *id = NULL;

    if (!_this || !_this->internal) {
        SDL_SetError("Video driver uninitialized");
    } else {
        driver_data = _this->internal;

        if (driver_data->data_device_manager) {
            id = wl_data_device_manager_create_data_source(
                driver_data->data_device_manager);
        }

        if (!id) {
            SDL_SetError("Wayland unable to create data source");
        } else {
            data_source = SDL_calloc(1, sizeof(*data_source));
            if (!data_source) {
                wl_data_source_destroy(id);
            } else {
                data_source->source = id;
                wl_data_source_set_user_data(id, data_source);
                wl_data_source_add_listener(id, &data_source_listener,
                                            data_source);
            }
        }
    }
    return data_source;
}

SDL_WaylandPrimarySelectionSource *Wayland_primary_selection_source_create(SDL_VideoDevice *_this)
{
    SDL_WaylandPrimarySelectionSource *primary_selection_source = NULL;
    SDL_VideoData *driver_data = NULL;
    struct zwp_primary_selection_source_v1 *id = NULL;

    if (!_this || !_this->internal) {
        SDL_SetError("Video driver uninitialized");
    } else {
        driver_data = _this->internal;

        if (driver_data->primary_selection_device_manager) {
            id = zwp_primary_selection_device_manager_v1_create_source(
                driver_data->primary_selection_device_manager);
        }

        if (!id) {
            SDL_SetError("Wayland unable to create primary selection source");
        } else {
            primary_selection_source = SDL_calloc(1, sizeof(*primary_selection_source));
            if (!primary_selection_source) {
                zwp_primary_selection_source_v1_destroy(id);
            } else {
                primary_selection_source->source = id;
                zwp_primary_selection_source_v1_add_listener(id, &primary_selection_source_listener,
                                                             primary_selection_source);
            }
        }
    }
    return primary_selection_source;
}

static void data_offer_handle_offer(void *data, struct wl_data_offer *wl_data_offer,
                                    const char *mime_type)
{
    SDL_WaylandDataOffer *offer = data;
    Wayland_data_offer_add_mime(offer, mime_type);
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In wl_data_offer_listener . data_offer_handle_offer on data_offer 0x%08x for MIME '%s'",
                 (wl_data_offer ? WAYLAND_wl_proxy_get_id((struct wl_proxy *)wl_data_offer) : -1),
                 mime_type);
}

static void data_offer_handle_source_actions(void *data, struct wl_data_offer *wl_data_offer,
                                             uint32_t source_actions)
{
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In wl_data_offer_listener . data_offer_handle_source_actions on data_offer 0x%08x for Source Actions '%d'",
                 (wl_data_offer ? WAYLAND_wl_proxy_get_id((struct wl_proxy *)wl_data_offer) : -1),
                 source_actions);
}

static void data_offer_handle_actions(void *data, struct wl_data_offer *wl_data_offer,
                                      uint32_t dnd_action)
{
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In wl_data_offer_listener . data_offer_handle_actions on data_offer 0x%08x for DND Actions '%d'",
                 (wl_data_offer ? WAYLAND_wl_proxy_get_id((struct wl_proxy *)wl_data_offer) : -1),
                 dnd_action);
}

static const struct wl_data_offer_listener data_offer_listener = {
    data_offer_handle_offer,
    data_offer_handle_source_actions, // Version 3
    data_offer_handle_actions,        // Version 3
};

static void primary_selection_offer_handle_offer(void *data, struct zwp_primary_selection_offer_v1 *zwp_primary_selection_offer_v1,
                                                 const char *mime_type)
{
    SDL_WaylandPrimarySelectionOffer *offer = data;
    Wayland_primary_selection_offer_add_mime(offer, mime_type);
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In zwp_primary_selection_offer_v1_listener . primary_selection_offer_handle_offer on primary_selection_offer 0x%08x for MIME '%s'",
                 (zwp_primary_selection_offer_v1 ? WAYLAND_wl_proxy_get_id((struct wl_proxy *)zwp_primary_selection_offer_v1) : -1),
                 mime_type);
}

static const struct zwp_primary_selection_offer_v1_listener primary_selection_offer_listener = {
    primary_selection_offer_handle_offer,
};

static void data_device_handle_data_offer(void *data, struct wl_data_device *wl_data_device,
                                          struct wl_data_offer *id)
{
    SDL_WaylandDataOffer *data_offer = SDL_calloc(1, sizeof(*data_offer));
    if (data_offer) {
        SDL_WaylandDataDevice *data_device = (SDL_WaylandDataDevice *)data;
        data_device->seat->display->last_incoming_data_offer_seat = data_device->seat;
        data_offer->offer = id;
        data_offer->data_device = data_device;
        data_offer->read_fd = -1;
        WAYLAND_wl_list_init(&(data_offer->mimes));
        wl_data_offer_set_user_data(id, data_offer);
        wl_data_offer_add_listener(id, &data_offer_listener, data_offer);
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In wl_data_device_listener . data_device_handle_data_offer on data_offer 0x%08x",
                     (id ? WAYLAND_wl_proxy_get_id((struct wl_proxy *)id) : -1));
    }
}

static void data_device_handle_enter(void *data, struct wl_data_device *wl_data_device,
                                     uint32_t serial, struct wl_surface *surface,
                                     wl_fixed_t x, wl_fixed_t y, struct wl_data_offer *id)
{
    SDL_WaylandDataDevice *data_device = data;
    data_device->has_mime_file = false;
    data_device->has_mime_text = false;
    uint32_t dnd_action = WL_DATA_DEVICE_MANAGER_DND_ACTION_NONE;

    data_device->drag_serial = serial;

    if (id) {
        data_device->drag_offer = wl_data_offer_get_user_data(id);

        // TODO: SDL Support more mime types
#ifdef SDL_USE_LIBDBUS
        if (Wayland_data_offer_has_mime(data_device->drag_offer, FILE_PORTAL_MIME)) {
            data_device->has_mime_file = true;
            data_device->mime_type = FILE_PORTAL_MIME;
            wl_data_offer_accept(id, serial, FILE_PORTAL_MIME);
        }
#endif
        if (Wayland_data_offer_has_mime(data_device->drag_offer, FILE_MIME)) {
            data_device->has_mime_file = true;
            data_device->mime_type = FILE_MIME;
            wl_data_offer_accept(id, serial, FILE_MIME);
        }

        size_t mime_count = 0;
        const char **text_mime_types = Wayland_GetTextMimeTypes(SDL_GetVideoDevice(), &mime_count);
        for (size_t i = 0; i < mime_count; ++i) {
            if (Wayland_data_offer_has_mime(data_device->drag_offer, text_mime_types[i])) {
                data_device->has_mime_text = true;
                data_device->mime_type = text_mime_types[i];
                wl_data_offer_accept(id, serial, text_mime_types[i]);
                break;
            }
        }

        // SDL only supports "copy" style drag and drop
        if (data_device->has_mime_file || data_device->has_mime_text) {
            dnd_action = WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY;
        } else {
            // drag_mime is NULL this will decline the offer
            wl_data_offer_accept(id, serial, NULL);
        }
        if (wl_data_offer_get_version(data_device->drag_offer->offer) >=
            WL_DATA_OFFER_SET_ACTIONS_SINCE_VERSION) {
            wl_data_offer_set_actions(data_device->drag_offer->offer,
                                      dnd_action, dnd_action);
        }

        // find the current window
        if (surface) {
            SDL_WindowData *window = Wayland_GetWindowDataForOwnedSurface(surface);
            if (window) {
                data_device->dnd_window = window->sdlwindow;
                const float dx = (float)wl_fixed_to_double(x);
                const float dy = (float)wl_fixed_to_double(y);
                SDL_SendDropPosition(data_device->dnd_window, dx, dy);
                SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                             ". In wl_data_device_listener . data_device_handle_enter on data_offer 0x%08x at %d x %d into window %d for serial %d",
                             WAYLAND_wl_proxy_get_id((struct wl_proxy *)id),
                             wl_fixed_to_int(x), wl_fixed_to_int(y), SDL_GetWindowID(data_device->dnd_window), serial);
            } else {
                data_device->dnd_window = NULL;
                SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                             ". In wl_data_device_listener . data_device_handle_enter on data_offer 0x%08x at %d x %d for serial %d",
                             WAYLAND_wl_proxy_get_id((struct wl_proxy *)id),
                             wl_fixed_to_int(x), wl_fixed_to_int(y), serial);
            }
        } else {
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In wl_data_device_listener . data_device_handle_enter on data_offer 0x%08x at %d x %d for serial %d",
                         WAYLAND_wl_proxy_get_id((struct wl_proxy *)id),
                         wl_fixed_to_int(x), wl_fixed_to_int(y), serial);
        }
    } else {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In wl_data_device_listener . data_device_handle_enter on data_offer 0x%08x at %d x %d for serial %d",
                     -1, wl_fixed_to_int(x), wl_fixed_to_int(y), serial);
    }
}

static void data_device_handle_leave(void *data, struct wl_data_device *wl_data_device)
{
    SDL_WaylandDataDevice *data_device = data;

    if (data_device->drag_offer) {
        if (data_device->dnd_window) {
            SDL_SendDropComplete(data_device->dnd_window);
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In wl_data_device_listener . data_device_handle_leave on data_offer 0x%08x from window %d for serial %d",
                         WAYLAND_wl_proxy_get_id((struct wl_proxy *)data_device->drag_offer->offer),
                         SDL_GetWindowID(data_device->dnd_window), data_device->drag_serial);
        } else {
            SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                         ". In wl_data_device_listener . data_device_handle_leave on data_offer 0x%08x for serial %d",
                         WAYLAND_wl_proxy_get_id((struct wl_proxy *)data_device->drag_offer->offer),
                         data_device->drag_serial);
        }
        Wayland_data_offer_destroy(data_device->drag_offer);
        data_device->drag_offer = NULL;
    } else {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In wl_data_device_listener . data_device_handle_leave on data_offer 0x%08x for serial %d",
                     -1, -1);
    }
    data_device->has_mime_file = false;
    data_device->has_mime_text = false;
}

static void data_device_handle_motion(void *data, struct wl_data_device *wl_data_device,
                                      uint32_t time, wl_fixed_t x, wl_fixed_t y)
{
    SDL_WaylandDataDevice *data_device = data;

    if (data_device->drag_offer && data_device->dnd_window && (data_device->has_mime_file || data_device->has_mime_text)) {
        const float dx = (float)wl_fixed_to_double(x);
        const float dy = (float)wl_fixed_to_double(y);

        /* XXX: Send the filename here if the event system ever starts passing it though.
         *      Any future implementation should cache the filenames, as otherwise this could
         *      hammer the DBus interface hundreds or even thousands of times per second.
         */
        SDL_SendDropPosition(data_device->dnd_window, dx, dy);
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In wl_data_device_listener . data_device_handle_motion on data_offer 0x%08x at %d x %d in window %d serial %d",
                     WAYLAND_wl_proxy_get_id((struct wl_proxy *)data_device->drag_offer->offer),
                     wl_fixed_to_int(x), wl_fixed_to_int(y),
                     SDL_GetWindowID(data_device->dnd_window), data_device->drag_serial);
    } else {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In wl_data_device_listener . data_device_handle_motion on data_offer 0x%08x at %d x %d serial %d",
                     -1, wl_fixed_to_int(x), wl_fixed_to_int(y), -1);
    }
}

static void data_device_handle_drop(void *data, struct wl_data_device *wl_data_device)
{
    SDL_WaylandDataDevice *data_device = data;

    if (data_device->drag_offer && data_device->dnd_window && (data_device->has_mime_file || data_device->has_mime_text)) {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In wl_data_device_listener . data_device_handle_drop on data_offer 0x%08x in window %d serial %d",
                     WAYLAND_wl_proxy_get_id((struct wl_proxy *)data_device->drag_offer->offer),
                     SDL_GetWindowID(data_device->dnd_window), data_device->drag_serial);
        // TODO: SDL Support more mime types
        size_t length;
        bool drop_handled = false;
#ifdef SDL_USE_LIBDBUS
        if (Wayland_data_offer_has_mime(data_device->drag_offer, FILE_PORTAL_MIME)) {
            void *buffer = Wayland_data_offer_receive(data_device->drag_offer,
                                                      FILE_PORTAL_MIME, &length);
            if (buffer) {
                SDL_DBusContext *dbus = SDL_DBus_GetContext();
                if (dbus) {
                    int path_count = 0;
                    char **paths = SDL_DBus_DocumentsPortalRetrieveFiles(buffer, &path_count);
                    // If dropped files contain a directory the list is empty
                    if (paths && path_count > 0) {
                        int i;
                        for (i = 0; i < path_count; i++) {
                            SDL_SendDropFile(data_device->dnd_window, NULL, paths[i]);
                        }
                        dbus->free_string_array(paths);
                        SDL_SendDropComplete(data_device->dnd_window);
                        drop_handled = true;
                    }
                }
                SDL_free(buffer);
            }
        }
#endif
        /* If XDG document portal fails fallback.
         * When running a flatpak sandbox this will most likely be a list of
         * non paths that are not visible to the application
         */
        if (!drop_handled) {
            void *buffer = Wayland_data_offer_receive(data_device->drag_offer, data_device->mime_type, &length);
            if (data_device->has_mime_file) {
                if (buffer) {
                    char *saveptr = NULL;
                    char *token = SDL_strtok_r((char *)buffer, "\r\n", &saveptr);
                    while (token) {
                        if (SDL_URIToLocal(token, token) >= 0) {
                            SDL_SendDropFile(data_device->dnd_window, NULL, token);
                        }
                        token = SDL_strtok_r(NULL, "\r\n", &saveptr);
                    }
                    SDL_free(buffer);
                    SDL_SendDropComplete(data_device->dnd_window);
                } else {
                    SDL_SendDropComplete(data_device->dnd_window);
                }
                drop_handled = true;
            } else if (data_device->has_mime_text) {
                if (buffer) {
                    char *saveptr = NULL;
                    char *token = SDL_strtok_r((char *)buffer, "\r\n", &saveptr);
                    while (token) {
                        SDL_SendDropText(data_device->dnd_window, token);
                        token = SDL_strtok_r(NULL, "\r\n", &saveptr);
                    }
                    SDL_free(buffer);
                    SDL_SendDropComplete(data_device->dnd_window);
                } else {
                    /* Even though there has been a valid data offer,
                     *  and there have been valid Enter, Motion, and Drop callbacks,
                     *  Wayland_data_offer_receive may return an empty buffer,
                     *  because the data is actually in the primary selection device,
                     *  not in the data device.
                     */
                    SDL_SendDropComplete(data_device->dnd_window);
                }
                drop_handled = true;
            }
        }

        if (drop_handled && wl_data_offer_get_version(data_device->drag_offer->offer) >= WL_DATA_OFFER_FINISH_SINCE_VERSION) {
            wl_data_offer_finish(data_device->drag_offer->offer);
        }
    } else {
        SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                     ". In wl_data_device_listener . data_device_handle_drop on data_offer 0x%08x serial %d",
                     -1, -1);
    }

    Wayland_data_offer_destroy(data_device->drag_offer);
    data_device->drag_offer = NULL;
}

static void data_device_handle_selection(void *data, struct wl_data_device *wl_data_device,
                                         struct wl_data_offer *id)
{
    SDL_WaylandDataDevice *data_device = data;
    SDL_WaylandDataOffer *offer = NULL;

    if (id) {
        offer = wl_data_offer_get_user_data(id);
    }

    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In data_device_listener . data_device_handle_selection on data_offer 0x%08x",
                 (id ? WAYLAND_wl_proxy_get_id((struct wl_proxy *)id) : -1));
    if (data_device->selection_offer != offer) {
        Wayland_data_offer_destroy(data_device->selection_offer);
        data_device->selection_offer = offer;
    }

    Wayland_data_offer_notify_from_mimes(offer, true);
}

static const struct wl_data_device_listener data_device_listener = {
    data_device_handle_data_offer,
    data_device_handle_enter,
    data_device_handle_leave,
    data_device_handle_motion,
    data_device_handle_drop,
    data_device_handle_selection
};

static void primary_selection_device_handle_offer(void *data, struct zwp_primary_selection_device_v1 *zwp_primary_selection_device_v1,
                                                  struct zwp_primary_selection_offer_v1 *id)
{
    SDL_WaylandPrimarySelectionOffer *primary_selection_offer = SDL_calloc(1, sizeof(*primary_selection_offer));
    if (primary_selection_offer) {
        SDL_WaylandPrimarySelectionDevice *primary_selection_device = (SDL_WaylandPrimarySelectionDevice *)data;
        primary_selection_device->seat->display->last_incoming_primary_selection_seat = primary_selection_device->seat;
        primary_selection_offer->offer = id;
        primary_selection_offer->primary_selection_device = primary_selection_device;
        WAYLAND_wl_list_init(&(primary_selection_offer->mimes));
        zwp_primary_selection_offer_v1_set_user_data(id, primary_selection_offer);
        zwp_primary_selection_offer_v1_add_listener(id, &primary_selection_offer_listener, primary_selection_offer);
    }
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In zwp_primary_selection_device_v1_listener . primary_selection_device_handle_offer on primary_selection_offer 0x%08x",
                 (id ? WAYLAND_wl_proxy_get_id((struct wl_proxy *)id) : -1));
}

static void primary_selection_device_handle_selection(void *data, struct zwp_primary_selection_device_v1 *zwp_primary_selection_device_v1,
                                                      struct zwp_primary_selection_offer_v1 *id)
{
    SDL_WaylandPrimarySelectionDevice *primary_selection_device = data;
    SDL_WaylandPrimarySelectionOffer *offer = NULL;

    if (id) {
        offer = zwp_primary_selection_offer_v1_get_user_data(id);
    }

    if (primary_selection_device->selection_offer != offer) {
        Wayland_primary_selection_offer_destroy(primary_selection_device->selection_offer);
        primary_selection_device->selection_offer = offer;
    }
    SDL_LogTrace(SDL_LOG_CATEGORY_INPUT,
                 ". In zwp_primary_selection_device_v1_listener . primary_selection_device_handle_selection on primary_selection_offer 0x%08x",
                 (id ? WAYLAND_wl_proxy_get_id((struct wl_proxy *)id) : -1));
}

static const struct zwp_primary_selection_device_v1_listener primary_selection_device_listener = {
    primary_selection_device_handle_offer,
    primary_selection_device_handle_selection
};

static void text_input_enter(void *data,
                             struct zwp_text_input_v3 *zwp_text_input_v3,
                             struct wl_surface *surface)
{
    // No-op
}

static void text_input_leave(void *data,
                             struct zwp_text_input_v3 *zwp_text_input_v3,
                             struct wl_surface *surface)
{
    // No-op
}

static void text_input_preedit_string(void *data,
                                      struct zwp_text_input_v3 *zwp_text_input_v3,
                                      const char *text,
                                      int32_t cursor_begin,
                                      int32_t cursor_end)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    seat->text_input.has_preedit = true;
    if (text) {
        int cursor_begin_utf8 = cursor_begin >= 0 ? (int)SDL_utf8strnlen(text, cursor_begin) : -1;
        int cursor_end_utf8 = cursor_end >= 0 ? (int)SDL_utf8strnlen(text, cursor_end) : -1;
        int cursor_size_utf8;
        if (cursor_end_utf8 >= 0) {
            if (cursor_begin_utf8 >= 0) {
                cursor_size_utf8 = cursor_end_utf8 - cursor_begin_utf8;
            } else {
                cursor_size_utf8 = cursor_end_utf8;
            }
        } else {
            cursor_size_utf8 = -1;
        }
        SDL_SendEditingText(text, cursor_begin_utf8, cursor_size_utf8);
    } else {
        SDL_SendEditingText("", 0, 0);
    }
}

static void text_input_commit_string(void *data,
                                     struct zwp_text_input_v3 *zwp_text_input_v3,
                                     const char *text)
{
    SDL_SendKeyboardText(text);
}

static void text_input_delete_surrounding_text(void *data,
                                               struct zwp_text_input_v3 *zwp_text_input_v3,
                                               uint32_t before_length,
                                               uint32_t after_length)
{
    // FIXME: Do we care about this event?
}

static void text_input_done(void *data,
                            struct zwp_text_input_v3 *zwp_text_input_v3,
                            uint32_t serial)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    if (!seat->text_input.has_preedit) {
        SDL_SendEditingText("", 0, 0);
    }
    seat->text_input.has_preedit = false;
}

static const struct zwp_text_input_v3_listener text_input_listener = {
    text_input_enter,
    text_input_leave,
    text_input_preedit_string,
    text_input_commit_string,
    text_input_delete_surrounding_text,
    text_input_done
};

static void Wayland_DataDeviceSetID(SDL_WaylandDataDevice *data_device)
{
    if (!data_device->id_str)
#ifdef SDL_USE_LIBDBUS
    {
        SDL_DBusContext *dbus = SDL_DBus_GetContext();
        if (dbus) {
            const char *id = dbus->bus_get_unique_name(dbus->session_conn);
            if (id) {
                data_device->id_str = SDL_strdup(id);
            }
        }
    }
    if (!data_device->id_str)
#endif
    {
        char id[24];
        Uint64 pid = (Uint64)getpid();
        SDL_snprintf(id, sizeof(id), "%" SDL_PRIu64, pid);
        data_device->id_str = SDL_strdup(id);
    }
}

static void Wayland_SeatCreateDataDevice(SDL_WaylandSeat *seat)
{
    if (!seat->display->data_device_manager) {
        return;
    }

    SDL_WaylandDataDevice *data_device = SDL_calloc(1, sizeof(*data_device));
    if (!data_device) {
        return;
    }

    data_device->data_device = wl_data_device_manager_get_data_device(
        seat->display->data_device_manager, seat->wl_seat);
    data_device->seat = seat;

    if (!data_device->data_device) {
        SDL_free(data_device);
    } else {
        Wayland_DataDeviceSetID(data_device);
        wl_data_device_set_user_data(data_device->data_device, data_device);
        wl_data_device_add_listener(data_device->data_device,
                                    &data_device_listener, data_device);
        seat->data_device = data_device;
    }
}

void Wayland_DisplayInitDataDeviceManager(SDL_VideoData *display)
{
    SDL_WaylandSeat *seat;
    wl_list_for_each (seat, &display->seat_list, link) {
        Wayland_SeatCreateDataDevice(seat);
    }
}

static void Wayland_SeatCreatePrimarySelectionDevice(SDL_WaylandSeat *seat)
{
    if (!seat->display->primary_selection_device_manager) {
        return;
    }

    SDL_WaylandPrimarySelectionDevice *primary_selection_device = SDL_calloc(1, sizeof(*primary_selection_device));
    if (!primary_selection_device) {
        return;
    }

    primary_selection_device->primary_selection_device = zwp_primary_selection_device_manager_v1_get_device(
        seat->display->primary_selection_device_manager, seat->wl_seat);
    primary_selection_device->seat = seat;

    if (!primary_selection_device->primary_selection_device) {
        SDL_free(primary_selection_device);
    } else {
        zwp_primary_selection_device_v1_set_user_data(primary_selection_device->primary_selection_device,
                                                      primary_selection_device);
        zwp_primary_selection_device_v1_add_listener(primary_selection_device->primary_selection_device,
                                                     &primary_selection_device_listener, primary_selection_device);
        seat->primary_selection_device = primary_selection_device;
    }
}

void Wayland_DisplayInitPrimarySelectionDeviceManager(SDL_VideoData *display)
{
    SDL_WaylandSeat *seat;
    wl_list_for_each (seat, &display->seat_list, link) {
        Wayland_SeatCreatePrimarySelectionDevice(seat);
    }
}

static void Wayland_SeatCreateTextInput(SDL_WaylandSeat *seat)
{
    if (seat->display->text_input_manager) {
        seat->text_input.zwp_text_input = zwp_text_input_manager_v3_get_text_input(seat->display->text_input_manager, seat->wl_seat);

        if (seat->text_input.zwp_text_input) {
            zwp_text_input_v3_set_user_data(seat->text_input.zwp_text_input, seat);
            zwp_text_input_v3_add_listener(seat->text_input.zwp_text_input,
                                           &text_input_listener, seat);
        }
    }
}

void Wayland_DisplayCreateTextInputManager(SDL_VideoData *d, uint32_t id)
{
#ifdef HAVE_FCITX
    const char *im_module = SDL_getenv("SDL_IM_MODULE");
    if (im_module && SDL_strcmp(im_module, "fcitx") == 0) {
        /* Override the Wayland text-input protocol when Fcitx is enabled, like how GTK_IM_MODULE does.
         *
         * The Fcitx wiki discourages enabling it under Wayland via SDL_IM_MODULE, so its presence must
         * be intentional, and this workaround is needed for fixing key repeat detection.
         */
        return;
    }
#endif

    d->text_input_manager = wl_registry_bind(d->registry, id, &zwp_text_input_manager_v3_interface, 1);

    SDL_WaylandSeat *seat;
    wl_list_for_each(seat, &d->seat_list, link) {
        Wayland_SeatCreateTextInput(seat);
    }
}

// Pen/Tablet support...

typedef struct SDL_WaylandPenTool  // a stylus, etc, on a tablet.
{
    SDL_PenID instance_id;
    SDL_PenInfo info;
    SDL_Window *tool_focus;
    struct zwp_tablet_tool_v2 *wltool;
    float x;
    float y;
    bool frame_motion_set;
    float frame_axes[SDL_PEN_AXIS_COUNT];
    Uint32 frame_axes_set;
    int frame_pen_down;
    int frame_buttons[3];
    struct wl_list link;
} SDL_WaylandPenTool;

static void tablet_tool_handle_type(void *data, struct zwp_tablet_tool_v2 *tool, uint32_t type)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    switch (type) {
        #define CASE(typ) case ZWP_TABLET_TOOL_V2_TYPE_##typ: sdltool->info.subtype = SDL_PEN_TYPE_##typ; return
        CASE(ERASER);
        CASE(PEN);
        CASE(PENCIL);
        CASE(AIRBRUSH);
        CASE(BRUSH);
        #undef CASE
        default: sdltool->info.subtype = SDL_PEN_TYPE_UNKNOWN;  // we'll decline to add this when the `done` event comes through.
    }
}

static void tablet_tool_handle_hardware_serial(void *data, struct zwp_tablet_tool_v2 *tool, uint32_t serial_hi, uint32_t serial_lo)
{
    // don't care about this atm.
}

static void tablet_tool_handle_hardware_id_wacom(void *data, struct zwp_tablet_tool_v2 *tool, uint32_t id_hi, uint32_t id_lo)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    sdltool->info.wacom_id = id_lo;
}

static void tablet_tool_handle_capability(void *data, struct zwp_tablet_tool_v2 *tool, uint32_t capability)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    switch (capability) {
        #define CASE(wltyp,sdltyp) case ZWP_TABLET_TOOL_V2_CAPABILITY_##wltyp: sdltool->info.capabilities |= sdltyp; return
        CASE(TILT, SDL_PEN_CAPABILITY_XTILT | SDL_PEN_CAPABILITY_YTILT);
        CASE(PRESSURE, SDL_PEN_CAPABILITY_PRESSURE);
        CASE(DISTANCE, SDL_PEN_CAPABILITY_DISTANCE);
        CASE(ROTATION, SDL_PEN_CAPABILITY_ROTATION);
        CASE(SLIDER, SDL_PEN_CAPABILITY_SLIDER);
        #undef CASE
        default: break;  // unsupported here.
    }
}

static void tablet_tool_handle_done(void *data, struct zwp_tablet_tool_v2 *tool)
{
}

static void tablet_tool_handle_removed(void *data, struct zwp_tablet_tool_v2 *tool)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    if (sdltool->instance_id) {
        SDL_RemovePenDevice(0, sdltool->instance_id);
    }
    zwp_tablet_tool_v2_destroy(tool);
    WAYLAND_wl_list_remove(&sdltool->link);
    SDL_free(sdltool);
}

static void tablet_tool_handle_proximity_in(void *data, struct zwp_tablet_tool_v2 *tool, uint32_t serial, struct zwp_tablet_v2 *tablet, struct wl_surface *surface)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    SDL_WindowData *windowdata = surface ? Wayland_GetWindowDataForOwnedSurface(surface) : NULL;
    sdltool->tool_focus = windowdata ? windowdata->sdlwindow : NULL;

    SDL_assert(sdltool->instance_id == 0);  // shouldn't be added at this point.
    if (sdltool->info.subtype != SDL_PEN_TYPE_UNKNOWN) {   // don't tell SDL about it if we don't know its role.
        sdltool->instance_id = SDL_AddPenDevice(0, NULL, &sdltool->info, sdltool);
    }

    // According to the docs, this should be followed by a motion event, where we'll send our SDL events.
}

static void tablet_tool_handle_proximity_out(void *data, struct zwp_tablet_tool_v2 *tool)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    sdltool->tool_focus = NULL;

    if (sdltool->instance_id) {
        SDL_RemovePenDevice(0, sdltool->instance_id);
        sdltool->instance_id = 0;
    }
}

static void tablet_tool_handle_down(void *data, struct zwp_tablet_tool_v2 *tool, uint32_t serial)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    sdltool->frame_pen_down = 1;
}

static void tablet_tool_handle_up(void *data, struct zwp_tablet_tool_v2 *tool)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    sdltool->frame_pen_down = 0;
}

static void tablet_tool_handle_motion(void *data, struct zwp_tablet_tool_v2 *tool, wl_fixed_t sx_w, wl_fixed_t sy_w)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    SDL_Window *window = sdltool->tool_focus;
    if (window) {
        const SDL_WindowData *windowdata = window->internal;
        sdltool->x = (float)(wl_fixed_to_double(sx_w) * windowdata->pointer_scale.x);
        sdltool->y = (float)(wl_fixed_to_double(sy_w) * windowdata->pointer_scale.y);
        sdltool->frame_motion_set = true;
    }
}

static void tablet_tool_handle_pressure(void *data, struct zwp_tablet_tool_v2 *tool, uint32_t pressure)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    sdltool->frame_axes[SDL_PEN_AXIS_PRESSURE] = ((float) pressure) / 65535.0f;
    sdltool->frame_axes_set |= (1u << SDL_PEN_AXIS_PRESSURE);
    if (pressure) {
        sdltool->frame_axes[SDL_PEN_AXIS_DISTANCE] = 0.0f;
        sdltool->frame_axes_set |= (1u << SDL_PEN_AXIS_DISTANCE);
    }
}

static void tablet_tool_handle_distance(void *data, struct zwp_tablet_tool_v2 *tool, uint32_t distance)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    sdltool->frame_axes[SDL_PEN_AXIS_DISTANCE] = ((float) distance) / 65535.0f;
    sdltool->frame_axes_set |= (1u << SDL_PEN_AXIS_DISTANCE);
    if (distance) {
        sdltool->frame_axes[SDL_PEN_AXIS_PRESSURE] = 0.0f;
        sdltool->frame_axes_set |= (1u << SDL_PEN_AXIS_PRESSURE);
    }
}

static void tablet_tool_handle_tilt(void *data, struct zwp_tablet_tool_v2 *tool, wl_fixed_t xtilt, wl_fixed_t ytilt)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    sdltool->frame_axes[SDL_PEN_AXIS_XTILT] = (float)(wl_fixed_to_double(xtilt));
    sdltool->frame_axes[SDL_PEN_AXIS_YTILT] = (float)(wl_fixed_to_double(ytilt));
    sdltool->frame_axes_set |= (1u << SDL_PEN_AXIS_XTILT) | (1u << SDL_PEN_AXIS_YTILT);
}

static void tablet_tool_handle_button(void *data, struct zwp_tablet_tool_v2 *tool, uint32_t serial, uint32_t button, uint32_t state)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    int sdlbutton;

    switch (button) {
    // see %{_includedir}/linux/input-event-codes.h
    case 0x14b: // BTN_STYLUS
        sdlbutton = 1;
        break;
    case 0x14c: // BTN_STYLUS2
        sdlbutton = 2;
        break;
    case 0x149: // BTN_STYLUS3
        sdlbutton = 3;
        break;
    default:
        return;  // don't care about this button, I guess.
    }

    SDL_assert((sdlbutton >= 1) && (sdlbutton <= SDL_arraysize(sdltool->frame_buttons)));
    sdltool->frame_buttons[sdlbutton-1] = (state == ZWP_TABLET_PAD_V2_BUTTON_STATE_PRESSED) ? 1 : 0;
}

static void tablet_tool_handle_rotation(void *data, struct zwp_tablet_tool_v2 *tool, wl_fixed_t degrees)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    const float rotation = (float)(wl_fixed_to_double(degrees));
    sdltool->frame_axes[SDL_PEN_AXIS_ROTATION] = (rotation > 180.0f) ? (rotation - 360.0f) : rotation;  // map to -180.0f ... 179.0f range
    sdltool->frame_axes_set |= (1u << SDL_PEN_AXIS_ROTATION);
}

static void tablet_tool_handle_slider(void *data, struct zwp_tablet_tool_v2 *tool, int32_t position)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;
    sdltool->frame_axes[SDL_PEN_AXIS_SLIDER] = position / 65535.f;
    sdltool->frame_axes_set |= (1u << SDL_PEN_AXIS_SLIDER);
}

static void tablet_tool_handle_wheel(void *data, struct zwp_tablet_tool_v2 *tool, int32_t degrees, int32_t clicks)
{
    // not supported at the moment
}

static void tablet_tool_handle_frame(void *data, struct zwp_tablet_tool_v2 *tool, uint32_t time)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) data;

    if (!sdltool->instance_id) {
        return;  // Not a pen we report on.
    }

    const Uint64 timestamp = Wayland_AdjustEventTimestampBase(Wayland_EventTimestampMSToNS(time));
    const SDL_PenID instance_id = sdltool->instance_id;
    SDL_Window *window = sdltool->tool_focus;

    // I don't know if this is necessary (or makes sense), but send motion before pen downs, but after pen ups, so you don't get unexpected lines drawn.
    if (sdltool->frame_motion_set && (sdltool->frame_pen_down != -1)) {
        if (sdltool->frame_pen_down) {
            SDL_SendPenMotion(timestamp, instance_id, window, sdltool->x, sdltool->y);
            SDL_SendPenTouch(timestamp, instance_id, window, false, true);  // !!! FIXME: how do we know what tip is in use?
        } else {
            SDL_SendPenTouch(timestamp, instance_id, window, false, false); // !!! FIXME: how do we know what tip is in use?
            SDL_SendPenMotion(timestamp, instance_id, window, sdltool->x, sdltool->y);
        }
    } else {
        if (sdltool->frame_pen_down != -1) {
            SDL_SendPenTouch(timestamp, instance_id, window, false, (sdltool->frame_pen_down != 0));  // !!! FIXME: how do we know what tip is in use?
        }

        if (sdltool->frame_motion_set) {
            SDL_SendPenMotion(timestamp, instance_id, window, sdltool->x, sdltool->y);
        }
    }

    for (SDL_PenAxis i = 0; i < SDL_PEN_AXIS_COUNT; i++) {
        if (sdltool->frame_axes_set & (1u << i)) {
            SDL_SendPenAxis(timestamp, instance_id, window, i, sdltool->frame_axes[i]);
        }
    }

    for (int i = 0; i < SDL_arraysize(sdltool->frame_buttons); i++) {
        const int state = sdltool->frame_buttons[i];
        if (state != -1) {
            SDL_SendPenButton(timestamp, instance_id, window, (Uint8)(i + 1), (state != 0));
            sdltool->frame_buttons[i] = -1;
        }
    }

    // reset for next frame.
    sdltool->frame_pen_down = -1;
    sdltool->frame_motion_set = false;
    sdltool->frame_axes_set = 0;
}

static const struct zwp_tablet_tool_v2_listener tablet_tool_listener = {
    tablet_tool_handle_type,
    tablet_tool_handle_hardware_serial,
    tablet_tool_handle_hardware_id_wacom,
    tablet_tool_handle_capability,
    tablet_tool_handle_done,
    tablet_tool_handle_removed,
    tablet_tool_handle_proximity_in,
    tablet_tool_handle_proximity_out,
    tablet_tool_handle_down,
    tablet_tool_handle_up,
    tablet_tool_handle_motion,
    tablet_tool_handle_pressure,
    tablet_tool_handle_distance,
    tablet_tool_handle_tilt,
    tablet_tool_handle_rotation,
    tablet_tool_handle_slider,
    tablet_tool_handle_wheel,
    tablet_tool_handle_button,
    tablet_tool_handle_frame
};


static void tablet_seat_handle_tablet_added(void *data, struct zwp_tablet_seat_v2 *zwp_tablet_seat_v2, struct zwp_tablet_v2 *tablet)
{
    // don't care atm.
}

static void tablet_seat_handle_tool_added(void *data, struct zwp_tablet_seat_v2 *zwp_tablet_seat_v2, struct zwp_tablet_tool_v2 *tool)
{
    SDL_WaylandSeat *seat = (SDL_WaylandSeat *)data;
    SDL_WaylandPenTool *sdltool = SDL_calloc(1, sizeof(*sdltool));

    if (sdltool) {  // if allocation failed, oh well, we won't report this device.
        sdltool->wltool = tool;
        sdltool->info.max_tilt = -1.0f;
        sdltool->info.num_buttons = -1;
        sdltool->frame_pen_down = -1;
        for (int i = 0; i < SDL_arraysize(sdltool->frame_buttons); i++) {
            sdltool->frame_buttons[i] = -1;
        }
        WAYLAND_wl_list_insert(&seat->tablet.tool_list, &sdltool->link);

        // this will send a bunch of zwp_tablet_tool_v2 events right up front to tell
        // us device details, with a "done" event to let us know we have everything.
        zwp_tablet_tool_v2_add_listener(tool, &tablet_tool_listener, sdltool);
    }
}

static void tablet_seat_handle_pad_added(void *data, struct zwp_tablet_seat_v2 *zwp_tablet_seat_v2, struct zwp_tablet_pad_v2 *pad)
{
    // we don't care atm.
}

static const struct zwp_tablet_seat_v2_listener tablet_seat_listener = {
    tablet_seat_handle_tablet_added,
    tablet_seat_handle_tool_added,
    tablet_seat_handle_pad_added
};

static void Wayland_SeatInitTabletSupport(SDL_WaylandSeat *seat)
{
    WAYLAND_wl_list_init(&seat->tablet.tool_list);
    seat->tablet.wl_tablet_seat = zwp_tablet_manager_v2_get_tablet_seat(seat->display->tablet_manager, seat->wl_seat);
    zwp_tablet_seat_v2_add_listener(seat->tablet.wl_tablet_seat, &tablet_seat_listener, seat);
}

void Wayland_DisplayInitTabletManager(SDL_VideoData *display)
{
    SDL_WaylandSeat *seat;
    wl_list_for_each (seat, &display->seat_list, link) {
        Wayland_SeatInitTabletSupport(seat);
    }
}

static void Wayland_remove_all_pens_callback(SDL_PenID instance_id, void *handle, void *userdata)
{
    SDL_WaylandPenTool *sdltool = (SDL_WaylandPenTool *) handle;
    zwp_tablet_tool_v2_destroy(sdltool->wltool);
    SDL_free(sdltool);
}

static void Wayland_SeatDestroyTablet(SDL_WaylandSeat *seat, bool send_events)
{
    if (send_events) {
        SDL_WaylandPenTool *pen, *temp;
        wl_list_for_each_safe (pen, temp, &seat->tablet.tool_list, link) {
            // Remove all tools for this seat, sending PROXIMITY_OUT events.
            tablet_tool_handle_removed(pen, pen->wltool);
        }
    } else {
        // Shutting down, just delete everything.
        SDL_RemoveAllPenDevices(Wayland_remove_all_pens_callback, NULL);
    }

    if (seat && seat->tablet.wl_tablet_seat) {
        zwp_tablet_seat_v2_destroy(seat->tablet.wl_tablet_seat);
        seat->tablet.wl_tablet_seat = NULL;
    }

    SDL_zero(seat->tablet);
    WAYLAND_wl_list_init(&seat->tablet.tool_list);
}

void Wayland_DisplayCreateSeat(SDL_VideoData *display, struct wl_seat *wl_seat, Uint32 id)
{
    SDL_WaylandSeat *seat = SDL_calloc(1, sizeof(SDL_WaylandSeat));
    if (!seat) {
        return;
    }

    // Keep the seats in the order in which they were added.
    WAYLAND_wl_list_insert(display->seat_list.prev, &seat->link);

    WAYLAND_wl_list_init(&seat->touch.points);
    seat->wl_seat = wl_seat;
    seat->display = display;
    seat->registry_id = id;
    seat->keyboard.xkb.current_group = XKB_GROUP_INVALID;

    Wayland_SeatCreateDataDevice(seat);
    Wayland_SeatCreatePrimarySelectionDevice(seat);
    Wayland_SeatCreateTextInput(seat);

    wl_seat_set_user_data(seat->wl_seat, seat);
    wl_seat_add_listener(seat->wl_seat, &seat_listener, seat);

    if (display->tablet_manager) {
        Wayland_SeatInitTabletSupport(seat);
    }

    WAYLAND_wl_display_flush(display->display);
}

void Wayland_SeatDestroy(SDL_WaylandSeat *seat, bool send_events)
{
    if (!seat) {
        return;
    }

    SDL_free(seat->name);

    if (seat->data_device) {
        Wayland_data_device_clear_selection(seat->data_device);
        if (seat->data_device->selection_offer) {
            Wayland_data_offer_destroy(seat->data_device->selection_offer);
        }
        if (seat->data_device->drag_offer) {
            Wayland_data_offer_destroy(seat->data_device->drag_offer);
        }
        if (seat->data_device->data_device) {
            if (wl_data_device_get_version(seat->data_device->data_device) >= WL_DATA_DEVICE_RELEASE_SINCE_VERSION) {
                wl_data_device_release(seat->data_device->data_device);
            } else {
                wl_data_device_destroy(seat->data_device->data_device);
            }
        }
        SDL_free(seat->data_device->id_str);
        SDL_free(seat->data_device);
    }

    if (seat->primary_selection_device) {
        if (seat->primary_selection_device->selection_offer) {
            Wayland_primary_selection_offer_destroy(seat->primary_selection_device->selection_offer);
        }
        if (seat->primary_selection_device->selection_source) {
            Wayland_primary_selection_source_destroy(seat->primary_selection_device->selection_source);
        }
        if (seat->primary_selection_device->primary_selection_device) {
            zwp_primary_selection_device_v1_destroy(seat->primary_selection_device->primary_selection_device);
        }
        SDL_free(seat->primary_selection_device);
    }

    if (seat->text_input.zwp_text_input) {
        zwp_text_input_v3_destroy(seat->text_input.zwp_text_input);
    }

    Wayland_SeatDestroyKeyboard(seat, send_events);
    Wayland_SeatDestroyPointer(seat, send_events);
    Wayland_SeatDestroyTouch(seat);
    Wayland_SeatDestroyTablet(seat, send_events);

    if (wl_seat_get_version(seat->wl_seat) >= WL_SEAT_RELEASE_SINCE_VERSION) {
        wl_seat_release(seat->wl_seat);
    } else {
        wl_seat_destroy(seat->wl_seat);
    }

    WAYLAND_wl_list_remove(&seat->link);
    SDL_free(seat);
}

bool Wayland_SeatHasRelativePointerFocus(SDL_WaylandSeat *seat)
{
    /* If a seat has both keyboard and pointer capabilities, relative focus will follow the keyboard
     * attached to that seat. Otherwise, relative focus will be gained if any other seat has keyboard
     * focus on the window with pointer focus.
     */
    if (seat->keyboard.wl_keyboard) {
        return seat->keyboard.focus && seat->keyboard.focus == seat->pointer.focus;
    } else {
        return seat->pointer.focus && seat->pointer.focus->keyboard_focus_count != 0;
    }
}

static void Wayland_SeatUpdateKeyboardGrab(SDL_WaylandSeat *seat)
{
    SDL_VideoData *display = seat->display;

    if (display->key_inhibitor_manager) {
        // Destroy the existing key inhibitor.
        if (seat->keyboard.key_inhibitor) {
            zwp_keyboard_shortcuts_inhibitor_v1_destroy(seat->keyboard.key_inhibitor);
            seat->keyboard.key_inhibitor = NULL;
        }

        if (seat->keyboard.wl_keyboard) {
            SDL_WindowData *w = seat->keyboard.focus;
            if (w) {
                SDL_Window *window = w->sdlwindow;

                // Don't grab the keyboard if it shouldn't be grabbed.
                if (window->flags & SDL_WINDOW_KEYBOARD_GRABBED) {
                    seat->keyboard.key_inhibitor =
                        zwp_keyboard_shortcuts_inhibit_manager_v1_inhibit_shortcuts(display->key_inhibitor_manager, w->surface, seat->wl_seat);
                }
            }
        }
    }
}

void Wayland_SeatUpdatePointerGrab(SDL_WaylandSeat *seat)
{
    SDL_VideoData *display = seat->display;

    if (display->pointer_constraints) {
        const bool has_relative_focus = Wayland_SeatHasRelativePointerFocus(seat);

        if (seat->pointer.locked_pointer && (!display->relative_mode_enabled || !has_relative_focus)) {
            zwp_locked_pointer_v1_destroy(seat->pointer.locked_pointer);
            seat->pointer.locked_pointer = NULL;

            // Update the cursor after destroying a relative move lock.
            Wayland_SeatUpdateCursor(seat);
        }

        if (seat->pointer.wl_pointer) {
            // If relative mode is active, and the pointer focus matches the keyboard focus, lock it.
            if (seat->display->relative_mode_enabled && has_relative_focus) {
                if (!seat->pointer.locked_pointer) {
                    // Creating a lock on a surface with an active confinement region on the same seat is a protocol error.
                    if (seat->pointer.confined_pointer) {
                        zwp_confined_pointer_v1_destroy(seat->pointer.confined_pointer);
                        seat->pointer.confined_pointer = NULL;
                    }

                    seat->pointer.locked_pointer = zwp_pointer_constraints_v1_lock_pointer(display->pointer_constraints,
                                                                                           seat->pointer.focus->surface,
                                                                                           seat->pointer.wl_pointer, NULL,
                                                                                           ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);
                    zwp_locked_pointer_v1_add_listener(seat->pointer.locked_pointer, &locked_pointer_listener, seat);

                    // Ensure that the relative pointer is hidden, if required.
                    Wayland_SeatUpdateCursor(seat);
                }

                // Locked the cursor for relative mode, nothing more to do.
                return;
            }

            /* A confine may already be active, in which case we should destroy it and create a new one
             * in case it changed size.
             */
            if (seat->pointer.confined_pointer) {
                zwp_confined_pointer_v1_destroy(seat->pointer.confined_pointer);
                seat->pointer.confined_pointer = NULL;
            }

            SDL_WindowData *w = seat->pointer.focus;
            if (!w) {
                return;
            }

            SDL_Window *window = w->sdlwindow;

            // Don't confine the pointer if the window doesn't have input focus, or it shouldn't be confined.
            if (!(window->flags & SDL_WINDOW_INPUT_FOCUS) ||
                (!(window->flags & SDL_WINDOW_MOUSE_GRABBED) && SDL_RectEmpty(&window->mouse_rect))) {
                return;
            }

            struct wl_region *confine_rect = NULL;
            if (!SDL_RectEmpty(&window->mouse_rect)) {
                SDL_Rect scaled_mouse_rect;
                Wayland_GetScaledMouseRect(window, &scaled_mouse_rect);

                /* Some compositors will only confine the pointer to an arbitrary region if the pointer
                 * is already within the confinement area when it is created.
                 */
                if (SDL_PointInRect(&seat->pointer.last_motion, &scaled_mouse_rect)) {
                    confine_rect = wl_compositor_create_region(display->compositor);
                    wl_region_add(confine_rect,
                                  scaled_mouse_rect.x,
                                  scaled_mouse_rect.y,
                                  scaled_mouse_rect.w,
                                  scaled_mouse_rect.h);
                } else {
                    /* Warp the pointer to the closest point within the confinement zone if outside,
                     * The confinement region will be created when a true position event is received.
                     */
                    int closest_x = seat->pointer.last_motion.x;
                    int closest_y = seat->pointer.last_motion.y;

                    if (closest_x < scaled_mouse_rect.x) {
                        closest_x = scaled_mouse_rect.x;
                    } else if (closest_x >= scaled_mouse_rect.x + scaled_mouse_rect.w) {
                        closest_x = (scaled_mouse_rect.x + scaled_mouse_rect.w) - 1;
                    }

                    if (closest_y < scaled_mouse_rect.y) {
                        closest_y = scaled_mouse_rect.y;
                    } else if (closest_y >= scaled_mouse_rect.y + scaled_mouse_rect.h) {
                        closest_y = (scaled_mouse_rect.y + scaled_mouse_rect.h) - 1;
                    }

                    Wayland_SeatWarpMouse(seat, w, closest_x, closest_y);
                }
            }

            if (confine_rect || (window->flags & SDL_WINDOW_MOUSE_GRABBED)) {
                seat->pointer.confined_pointer =
                    zwp_pointer_constraints_v1_confine_pointer(display->pointer_constraints,
                                                               w->surface,
                                                               seat->pointer.wl_pointer,
                                                               confine_rect,
                                                               ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);
                zwp_confined_pointer_v1_add_listener(seat->pointer.confined_pointer,
                                                     &confined_pointer_listener,
                                                     window);

                if (confine_rect) {
                    wl_region_destroy(confine_rect);
                }

                // Commit the new confinement region immediately.
                wl_surface_commit(w->surface);
            }
        }
    }
}

void Wayland_DisplayUpdatePointerGrabs(SDL_VideoData *display, SDL_WindowData *window)
{
    SDL_WaylandSeat *seat;
    wl_list_for_each (seat, &display->seat_list, link) {
        if (!window || seat->pointer.focus == window) {
            Wayland_SeatUpdatePointerGrab(seat);
        }
    }
}

void Wayland_DisplayUpdateKeyboardGrabs(SDL_VideoData *display, SDL_WindowData *window)
{
    SDL_WaylandSeat *seat;
    wl_list_for_each (seat, &display->seat_list, link) {
        if (!window || seat->keyboard.focus == window) {
            Wayland_SeatUpdateKeyboardGrab(seat);
        }
    }
}

void Wayland_UpdateImplicitGrabSerial(SDL_WaylandSeat *seat, Uint32 serial)
{
    if (serial > seat->last_implicit_grab_serial) {
        seat->last_implicit_grab_serial = serial;
        seat->display->last_implicit_grab_seat = seat;
        Wayland_data_device_set_serial(seat->data_device, serial);
        Wayland_primary_selection_device_set_serial(seat->primary_selection_device, serial);
    }
}

#endif // SDL_VIDEO_DRIVER_WAYLAND
