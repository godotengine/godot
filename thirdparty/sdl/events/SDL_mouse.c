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

// General mouse handling code for SDL

#include "../SDL_hints_c.h"
#include "../video/SDL_sysvideo.h"
#include "SDL_events_c.h"
#include "SDL_mouse_c.h"
#if defined(SDL_PLATFORM_WINDOWS)
#include "../core/windows/SDL_windows.h" // For GetDoubleClickTime()
#endif

// #define DEBUG_MOUSE

#define WARP_EMULATION_THRESHOLD_NS SDL_MS_TO_NS(30)

typedef struct SDL_MouseInstance
{
    SDL_MouseID instance_id;
    char *name;
} SDL_MouseInstance;

// The mouse state
static SDL_Mouse SDL_mouse;
static int SDL_mouse_count;
static SDL_MouseInstance *SDL_mice;

// for mapping mouse events to touch
static bool track_mouse_down = false;

static void SDL_PrivateSendMouseMotion(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, bool relative, float x, float y);

static void SDLCALL SDL_MouseDoubleClickTimeChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    if (hint && *hint) {
        mouse->double_click_time = SDL_atoi(hint);
    } else {
#if defined(SDL_PLATFORM_WIN32) || defined(SDL_PLATFORM_WINGDK)
        mouse->double_click_time = GetDoubleClickTime();
#else
        mouse->double_click_time = 500;
#endif
    }
}

static void SDLCALL SDL_MouseDoubleClickRadiusChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    if (hint && *hint) {
        mouse->double_click_radius = SDL_atoi(hint);
    } else {
        mouse->double_click_radius = 32; // 32 pixels seems about right for touch interfaces
    }
}

static void SDLCALL SDL_MouseNormalSpeedScaleChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    if (hint && *hint) {
        mouse->enable_normal_speed_scale = true;
        mouse->normal_speed_scale = (float)SDL_atof(hint);
    } else {
        mouse->enable_normal_speed_scale = false;
        mouse->normal_speed_scale = 1.0f;
    }
}

static void SDLCALL SDL_MouseRelativeSpeedScaleChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    if (hint && *hint) {
        mouse->enable_relative_speed_scale = true;
        mouse->relative_speed_scale = (float)SDL_atof(hint);
    } else {
        mouse->enable_relative_speed_scale = false;
        mouse->relative_speed_scale = 1.0f;
    }
}

static void SDLCALL SDL_MouseRelativeModeCenterChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->relative_mode_center = SDL_GetStringBoolean(hint, true);
}

static void SDLCALL SDL_MouseRelativeSystemScaleChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->enable_relative_system_scale = SDL_GetStringBoolean(hint, false);
}

static void SDLCALL SDL_MouseWarpEmulationChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->warp_emulation_hint = SDL_GetStringBoolean(hint, true);

    if (!mouse->warp_emulation_hint && mouse->warp_emulation_active) {
        SDL_SetRelativeMouseMode(false);
        mouse->warp_emulation_active = false;
    }
}

static void SDLCALL SDL_TouchMouseEventsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->touch_mouse_events = SDL_GetStringBoolean(hint, true);
}

#ifdef SDL_PLATFORM_VITA
static void SDLCALL SDL_VitaTouchMouseDeviceChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;
    if (hint) {
        switch (*hint) {
        default:
        case '0':
            mouse->vita_touch_mouse_device = 1;
            break;
        case '1':
            mouse->vita_touch_mouse_device = 2;
            break;
        case '2':
            mouse->vita_touch_mouse_device = 3;
            break;
        }
    }
}
#endif

static void SDLCALL SDL_MouseTouchEventsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;
    bool default_value;

#if defined(SDL_PLATFORM_ANDROID) || (defined(SDL_PLATFORM_IOS) && !defined(SDL_PLATFORM_TVOS))
    default_value = true;
#else
    default_value = false;
#endif
    mouse->mouse_touch_events = SDL_GetStringBoolean(hint, default_value);

    if (mouse->mouse_touch_events) {
        if (!mouse->added_mouse_touch_device) {
            SDL_AddTouch(SDL_MOUSE_TOUCHID, SDL_TOUCH_DEVICE_DIRECT, "mouse_input");
            mouse->added_mouse_touch_device = true;
        }
    } else {
        if (mouse->added_mouse_touch_device) {
            SDL_DelTouch(SDL_MOUSE_TOUCHID);
            mouse->added_mouse_touch_device = false;
        }
    }
}

static void SDLCALL SDL_PenMouseEventsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->pen_mouse_events = SDL_GetStringBoolean(hint, true);
}

static void SDLCALL SDL_PenTouchEventsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->pen_touch_events = SDL_GetStringBoolean(hint, true);

    if (mouse->pen_touch_events) {
        if (!mouse->added_pen_touch_device) {
            SDL_AddTouch(SDL_PEN_TOUCHID, SDL_TOUCH_DEVICE_DIRECT, "pen_input");
            mouse->added_pen_touch_device = true;
        }
    } else {
        if (mouse->added_pen_touch_device) {
            SDL_DelTouch(SDL_PEN_TOUCHID);
            mouse->added_pen_touch_device = false;
        }
    }
}

static void SDLCALL SDL_MouseAutoCaptureChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;
    bool auto_capture = SDL_GetStringBoolean(hint, true);

    if (auto_capture != mouse->auto_capture) {
        mouse->auto_capture = auto_capture;
        SDL_UpdateMouseCapture(false);
    }
}

static void SDLCALL SDL_MouseRelativeWarpMotionChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->relative_mode_warp_motion = SDL_GetStringBoolean(hint, false);
}

static void SDLCALL SDL_MouseRelativeCursorVisibleChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    mouse->relative_mode_hide_cursor = !(SDL_GetStringBoolean(hint, false));

    SDL_RedrawCursor(); // Update cursor visibility
}

static void SDLCALL SDL_MouseIntegerModeChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Mouse *mouse = (SDL_Mouse *)userdata;

    if (hint && *hint) {
        mouse->integer_mode_flags = (Uint8)SDL_atoi(hint);
    } else {
        mouse->integer_mode_flags = 0;
    }
}

// Public functions
bool SDL_PreInitMouse(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    SDL_zerop(mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_DOUBLE_CLICK_TIME,
                        SDL_MouseDoubleClickTimeChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_DOUBLE_CLICK_RADIUS,
                        SDL_MouseDoubleClickRadiusChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_NORMAL_SPEED_SCALE,
                        SDL_MouseNormalSpeedScaleChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_RELATIVE_SPEED_SCALE,
                        SDL_MouseRelativeSpeedScaleChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_RELATIVE_SYSTEM_SCALE,
                        SDL_MouseRelativeSystemScaleChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_RELATIVE_MODE_CENTER,
                        SDL_MouseRelativeModeCenterChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_EMULATE_WARP_WITH_RELATIVE,
                        SDL_MouseWarpEmulationChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_TOUCH_MOUSE_EVENTS,
                        SDL_TouchMouseEventsChanged, mouse);

#ifdef SDL_PLATFORM_VITA
    SDL_AddHintCallback(SDL_HINT_VITA_TOUCH_MOUSE_DEVICE,
                        SDL_VitaTouchMouseDeviceChanged, mouse);
#endif

    SDL_AddHintCallback(SDL_HINT_MOUSE_TOUCH_EVENTS,
                        SDL_MouseTouchEventsChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_PEN_MOUSE_EVENTS,
                        SDL_PenMouseEventsChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_PEN_TOUCH_EVENTS,
                        SDL_PenTouchEventsChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_AUTO_CAPTURE,
                        SDL_MouseAutoCaptureChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_RELATIVE_WARP_MOTION,
                        SDL_MouseRelativeWarpMotionChanged, mouse);

    SDL_AddHintCallback(SDL_HINT_MOUSE_RELATIVE_CURSOR_VISIBLE,
                        SDL_MouseRelativeCursorVisibleChanged, mouse);

    SDL_AddHintCallback("SDL_MOUSE_INTEGER_MODE",
                        SDL_MouseIntegerModeChanged, mouse);

    mouse->was_touch_mouse_events = false; // no touch to mouse movement event pending

    mouse->cursor_visible = true;

    return true;
}

void SDL_PostInitMouse(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    /* Create a dummy mouse cursor for video backends that don't support true cursors,
     * so that mouse grab and focus functionality will work.
     */
    if (!mouse->def_cursor) {
        SDL_Surface *surface = SDL_CreateSurface(1, 1, SDL_PIXELFORMAT_ARGB8888);
        if (surface) {
            SDL_memset(surface->pixels, 0, (size_t)surface->h * surface->pitch);
            SDL_SetDefaultCursor(SDL_CreateColorCursor(surface, 0, 0));
            SDL_DestroySurface(surface);
        }
    }
}

bool SDL_IsMouse(Uint16 vendor, Uint16 product)
{
    // Eventually we'll have a blacklist of devices that enumerate as mice but aren't really
    return true;
}

static int SDL_GetMouseIndex(SDL_MouseID mouseID)
{
    for (int i = 0; i < SDL_mouse_count; ++i) {
        if (mouseID == SDL_mice[i].instance_id) {
            return i;
        }
    }
    return -1;
}

void SDL_AddMouse(SDL_MouseID mouseID, const char *name, bool send_event)
{
    int mouse_index = SDL_GetMouseIndex(mouseID);
    if (mouse_index >= 0) {
        // We already know about this mouse
        return;
    }

    SDL_assert(mouseID != 0);

    SDL_MouseInstance *mice = (SDL_MouseInstance *)SDL_realloc(SDL_mice, (SDL_mouse_count + 1) * sizeof(*mice));
    if (!mice) {
        return;
    }
    SDL_MouseInstance *instance = &mice[SDL_mouse_count];
    instance->instance_id = mouseID;
    instance->name = SDL_strdup(name ? name : "");
    SDL_mice = mice;
    ++SDL_mouse_count;

    if (send_event) {
        SDL_Event event;
        SDL_zero(event);
        event.type = SDL_EVENT_MOUSE_ADDED;
        event.mdevice.which = mouseID;
        SDL_PushEvent(&event);
    }
}

void SDL_RemoveMouse(SDL_MouseID mouseID, bool send_event)
{
    int mouse_index = SDL_GetMouseIndex(mouseID);
    if (mouse_index < 0) {
        // We don't know about this mouse
        return;
    }

    SDL_free(SDL_mice[mouse_index].name);

    if (mouse_index != SDL_mouse_count - 1) {
        SDL_memmove(&SDL_mice[mouse_index], &SDL_mice[mouse_index + 1], (SDL_mouse_count - mouse_index - 1) * sizeof(SDL_mice[mouse_index]));
    }
    --SDL_mouse_count;

    // Remove any mouse input sources for this mouseID
    SDL_Mouse *mouse = SDL_GetMouse();
    for (int i = 0; i < mouse->num_sources; ++i) {
        SDL_MouseInputSource *source = &mouse->sources[i];
        if (source->mouseID == mouseID) {
            SDL_free(source->clickstate);
            if (i != mouse->num_sources - 1) {
                SDL_memmove(&mouse->sources[i], &mouse->sources[i + 1], (mouse->num_sources - i - 1) * sizeof(mouse->sources[i]));
            }
            --mouse->num_sources;
            break;
        }
    }

    if (send_event) {
        SDL_Event event;
        SDL_zero(event);
        event.type = SDL_EVENT_MOUSE_REMOVED;
        event.mdevice.which = mouseID;
        SDL_PushEvent(&event);
    }
}

void SDL_SetMouseName(SDL_MouseID mouseID, const char *name)
{
    SDL_assert(mouseID != 0);

    const int mouse_index = SDL_GetMouseIndex(mouseID);

    if (mouse_index >= 0) {
        SDL_MouseInstance *instance = &SDL_mice[mouse_index];
        SDL_free(instance->name);
        instance->name = SDL_strdup(name ? name : "");
    }
}

bool SDL_HasMouse(void)
{
    return (SDL_mouse_count > 0);
}

SDL_MouseID *SDL_GetMice(int *count)
{
    int i;
    SDL_MouseID *mice;

    mice = (SDL_JoystickID *)SDL_malloc((SDL_mouse_count + 1) * sizeof(*mice));
    if (mice) {
        if (count) {
            *count = SDL_mouse_count;
        }

        for (i = 0; i < SDL_mouse_count; ++i) {
            mice[i] = SDL_mice[i].instance_id;
        }
        mice[i] = 0;
    } else {
        if (count) {
            *count = 0;
        }
    }

    return mice;
}

const char *SDL_GetMouseNameForID(SDL_MouseID instance_id)
{
    int mouse_index = SDL_GetMouseIndex(instance_id);
    if (mouse_index < 0) {
        SDL_SetError("Mouse %" SDL_PRIu32 " not found", instance_id);
        return NULL;
    }
    return SDL_GetPersistentString(SDL_mice[mouse_index].name);
}

void SDL_SetDefaultCursor(SDL_Cursor *cursor)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (cursor == mouse->def_cursor) {
        return;
    }

    if (mouse->def_cursor) {
        SDL_Cursor *default_cursor = mouse->def_cursor;
        SDL_Cursor *prev, *curr;

        if (mouse->cur_cursor == mouse->def_cursor) {
            mouse->cur_cursor = NULL;
        }
        mouse->def_cursor = NULL;

        for (prev = NULL, curr = mouse->cursors; curr;
             prev = curr, curr = curr->next) {
            if (curr == default_cursor) {
                if (prev) {
                    prev->next = curr->next;
                } else {
                    mouse->cursors = curr->next;
                }

                break;
            }
        }

        if (mouse->FreeCursor && default_cursor->internal) {
            mouse->FreeCursor(default_cursor);
        } else {
            SDL_free(default_cursor);
        }
    }

    mouse->def_cursor = cursor;

    if (!mouse->cur_cursor) {
        SDL_SetCursor(cursor);
    }
}

SDL_SystemCursor SDL_GetDefaultSystemCursor(void)
{
    SDL_SystemCursor id = SDL_SYSTEM_CURSOR_DEFAULT;
    const char *value = SDL_GetHint(SDL_HINT_MOUSE_DEFAULT_SYSTEM_CURSOR);
    if (value) {
        int index = SDL_atoi(value);
        if (0 <= index && index < SDL_SYSTEM_CURSOR_COUNT) {
            id = (SDL_SystemCursor)index;
        }
    }
    return id;
}

SDL_Mouse *SDL_GetMouse(void)
{
    return &SDL_mouse;
}

static SDL_MouseButtonFlags SDL_GetMouseButtonState(SDL_Mouse *mouse, SDL_MouseID mouseID, bool include_touch)
{
    int i;
    SDL_MouseButtonFlags buttonstate = 0;

    for (i = 0; i < mouse->num_sources; ++i) {
        if (mouseID == SDL_GLOBAL_MOUSE_ID || mouseID == SDL_TOUCH_MOUSEID) {
            if (include_touch || mouse->sources[i].mouseID != SDL_TOUCH_MOUSEID) {
                buttonstate |= mouse->sources[i].buttonstate;
            }
        } else {
            if (mouseID == mouse->sources[i].mouseID) {
                buttonstate |= mouse->sources[i].buttonstate;
                break;
            }
        }
    }
    return buttonstate;
}

SDL_Window *SDL_GetMouseFocus(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    return mouse->focus;
}

/* TODO RECONNECT: Hello from the Wayland video driver!
 * This was once removed from SDL, but it's been added back in comment form
 * because we will need it when Wayland adds compositor reconnect support.
 * If you need this before we do, great! Otherwise, leave this alone, we'll
 * uncomment it at the right time.
 * -flibit
 */
#if 0
void SDL_ResetMouse(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    Uint32 buttonState = SDL_GetMouseButtonState(mouse, SDL_GLOBAL_MOUSE_ID, false);
    int i;

    for (i = 1; i <= sizeof(buttonState)*8; ++i) {
        if (buttonState & SDL_BUTTON_MASK(i)) {
            SDL_SendMouseButton(0, mouse->focus, mouse->mouseID, i, false);
        }
    }
    SDL_assert(SDL_GetMouseButtonState(mouse, SDL_GLOBAL_MOUSE_ID, false) == 0);
}
#endif // 0

void SDL_SetMouseFocus(SDL_Window *window)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->focus == window) {
        return;
    }

    /* Actually, this ends up being a bad idea, because most operating
       systems have an implicit grab when you press the mouse button down
       so you can drag things out of the window and then get the mouse up
       when it happens.  So, #if 0...
    */
#if 0
    if (mouse->focus && !window) {
        // We won't get anymore mouse messages, so reset mouse state
        SDL_ResetMouse();
    }
#endif

    // See if the current window has lost focus
    if (mouse->focus) {
        SDL_SendWindowEvent(mouse->focus, SDL_EVENT_WINDOW_MOUSE_LEAVE, 0, 0);
    }

    mouse->focus = window;
    mouse->has_position = false;

    if (mouse->focus) {
        SDL_SendWindowEvent(mouse->focus, SDL_EVENT_WINDOW_MOUSE_ENTER, 0, 0);
    }

    // Update cursor visibility
    SDL_RedrawCursor();
}

bool SDL_MousePositionInWindow(SDL_Window *window, float x, float y)
{
    if (!window) {
        return false;
    }

    if (window && !(window->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
        if (x < 0.0f || y < 0.0f || x >= (float)window->w || y >= (float)window->h) {
            return false;
        }
    }
    return true;
}

// Check to see if we need to synthesize focus events
static bool SDL_UpdateMouseFocus(SDL_Window *window, float x, float y, Uint32 buttonstate, bool send_mouse_motion)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    bool inWindow = SDL_MousePositionInWindow(window, x, y);

    if (!inWindow) {
        if (window == mouse->focus) {
#ifdef DEBUG_MOUSE
            SDL_Log("Mouse left window, synthesizing move & focus lost event");
#endif
            if (send_mouse_motion) {
                SDL_PrivateSendMouseMotion(0, window, SDL_GLOBAL_MOUSE_ID, false, x, y);
            }
            SDL_SetMouseFocus(NULL);
        }
        return false;
    }

    if (window != mouse->focus) {
#ifdef DEBUG_MOUSE
        SDL_Log("Mouse entered window, synthesizing focus gain & move event");
#endif
        SDL_SetMouseFocus(window);
        if (send_mouse_motion) {
            SDL_PrivateSendMouseMotion(0, window, SDL_GLOBAL_MOUSE_ID, false, x, y);
        }
    }
    return true;
}

void SDL_SendMouseMotion(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, bool relative, float x, float y)
{
    if (window && !relative) {
        SDL_Mouse *mouse = SDL_GetMouse();
        if (!SDL_UpdateMouseFocus(window, x, y, SDL_GetMouseButtonState(mouse, mouseID, true), (mouseID != SDL_TOUCH_MOUSEID && mouseID != SDL_PEN_MOUSEID))) {
            return;
        }
    }

    SDL_PrivateSendMouseMotion(timestamp, window, mouseID, relative, x, y);
}

static void ConstrainMousePosition(SDL_Mouse *mouse, SDL_Window *window, float *x, float *y)
{
    /* make sure that the pointers find themselves inside the windows,
       unless we have the mouse captured. */
    if (window && !(window->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
        int x_min = 0, x_max = window->w - 1;
        int y_min = 0, y_max = window->h - 1;
        const SDL_Rect *confine = SDL_GetWindowMouseRect(window);

        if (confine) {
            SDL_Rect window_rect;
            SDL_Rect mouse_rect;

            window_rect.x = 0;
            window_rect.y = 0;
            window_rect.w = x_max + 1;
            window_rect.h = y_max + 1;
            if (SDL_GetRectIntersection(confine, &window_rect, &mouse_rect)) {
                x_min = mouse_rect.x;
                y_min = mouse_rect.y;
                x_max = x_min + mouse_rect.w - 1;
                y_max = y_min + mouse_rect.h - 1;
            }
        }

        if (*x >= (float)(x_max + 1)) {
            *x = SDL_max((float)x_max, mouse->last_x);
        }
        if (*x < (float)x_min) {
            *x = (float)x_min;
        }

        if (*y >= (float)(y_max + 1)) {
            *y = SDL_max((float)y_max, mouse->last_y);
        }
        if (*y < (float)y_min) {
            *y = (float)y_min;
        }
    }
}

static void SDL_PrivateSendMouseMotion(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, bool relative, float x, float y)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    float xrel = 0.0f;
    float yrel = 0.0f;
    bool window_is_relative = mouse->focus && (mouse->focus->flags & SDL_WINDOW_MOUSE_RELATIVE_MODE);

    // SDL_HINT_MOUSE_TOUCH_EVENTS: controlling whether mouse events should generate synthetic touch events
    if (mouse->mouse_touch_events) {
        if (mouseID != SDL_TOUCH_MOUSEID && mouseID != SDL_PEN_MOUSEID && !relative && track_mouse_down) {
            if (window) {
                float normalized_x = x / (float)window->w;
                float normalized_y = y / (float)window->h;
                SDL_SendTouchMotion(timestamp, SDL_MOUSE_TOUCHID, SDL_BUTTON_LEFT, window, normalized_x, normalized_y, 1.0f);
            }
        }
    }

    // SDL_HINT_TOUCH_MOUSE_EVENTS: if not set, discard synthetic mouse events coming from platform layer
    if (!mouse->touch_mouse_events && mouseID == SDL_TOUCH_MOUSEID) {
        return;
    }

    if (relative) {
        if (mouse->relative_mode) {
            if (mouse->InputTransform) {
                void *data = mouse->input_transform_data;
                mouse->InputTransform(data, timestamp, window, mouseID, &x, &y);
            } else {
                if (mouse->enable_relative_system_scale) {
                    if (mouse->ApplySystemScale) {
                        mouse->ApplySystemScale(mouse->system_scale_data, timestamp, window, mouseID, &x, &y);
                    }
                }
                if (mouse->enable_relative_speed_scale) {
                    x *= mouse->relative_speed_scale;
                    y *= mouse->relative_speed_scale;
                }
            }
        } else {
            if (mouse->enable_normal_speed_scale) {
                x *= mouse->normal_speed_scale;
                y *= mouse->normal_speed_scale;
            }
        }
        if (mouse->integer_mode_flags & 1) {
            // Accumulate the fractional relative motion and only process the integer portion
            mouse->integer_mode_residual_motion_x = SDL_modff(mouse->integer_mode_residual_motion_x + x, &x);
            mouse->integer_mode_residual_motion_y = SDL_modff(mouse->integer_mode_residual_motion_y + y, &y);
        }
        xrel = x;
        yrel = y;
        x = (mouse->last_x + xrel);
        y = (mouse->last_y + yrel);
        ConstrainMousePosition(mouse, window, &x, &y);
    } else {
        if (mouse->integer_mode_flags & 1) {
            // Discard the fractional component from absolute coordinates
            x = SDL_truncf(x);
            y = SDL_truncf(y);
        }
        ConstrainMousePosition(mouse, window, &x, &y);
        if (mouse->has_position) {
            xrel = x - mouse->last_x;
            yrel = y - mouse->last_y;
        }
    }

    if (mouse->has_position && xrel == 0.0f && yrel == 0.0f) { // Drop events that don't change state
#ifdef DEBUG_MOUSE
        SDL_Log("Mouse event didn't change state - dropped!");
#endif
        return;
    }

    // Ignore relative motion positioning the first touch
    if (mouseID == SDL_TOUCH_MOUSEID && !SDL_GetMouseButtonState(mouse, mouseID, true)) {
        xrel = 0.0f;
        yrel = 0.0f;
    }

    // modify internal state
    {
        mouse->x_accu += xrel;
        mouse->y_accu += yrel;

        if (relative && mouse->has_position) {
            mouse->x += xrel;
            mouse->y += yrel;
            ConstrainMousePosition(mouse, window, &mouse->x, &mouse->y);
        } else {
            mouse->x = x;
            mouse->y = y;
        }
        mouse->has_position = true;

        // Use unclamped values if we're getting events outside the window
        mouse->last_x = relative ? mouse->x : x;
        mouse->last_y = relative ? mouse->y : y;

        mouse->click_motion_x += xrel;
        mouse->click_motion_y += yrel;
    }

    // Move the mouse cursor, if needed
    if (mouse->cursor_visible && !mouse->relative_mode &&
        mouse->MoveCursor && mouse->cur_cursor) {
        mouse->MoveCursor(mouse->cur_cursor);
    }

    // Post the event, if desired
    if (SDL_EventEnabled(SDL_EVENT_MOUSE_MOTION)) {
        if ((!mouse->relative_mode || mouse->warp_emulation_active) && mouseID != SDL_TOUCH_MOUSEID && mouseID != SDL_PEN_MOUSEID) {
            // We're not in relative mode, so all mouse events are global mouse events
            mouseID = SDL_GLOBAL_MOUSE_ID;
        }

        if (!relative && window_is_relative) {
            if (!mouse->relative_mode_warp_motion) {
                return;
            }
            xrel = 0.0f;
            yrel = 0.0f;
        }

        SDL_Event event;
        event.type = SDL_EVENT_MOUSE_MOTION;
        event.common.timestamp = timestamp;
        event.motion.windowID = mouse->focus ? mouse->focus->id : 0;
        event.motion.which = mouseID;
        // Set us pending (or clear during a normal mouse movement event) as having triggered
        mouse->was_touch_mouse_events = (mouseID == SDL_TOUCH_MOUSEID);
        event.motion.state = SDL_GetMouseButtonState(mouse, mouseID, true);
        event.motion.x = mouse->x;
        event.motion.y = mouse->y;
        event.motion.xrel = xrel;
        event.motion.yrel = yrel;
        SDL_PushEvent(&event);
    }
}

static SDL_MouseInputSource *GetMouseInputSource(SDL_Mouse *mouse, SDL_MouseID mouseID, bool down, Uint8 button)
{
    SDL_MouseInputSource *source, *match = NULL, *sources;
    int i;

    for (i = 0; i < mouse->num_sources; ++i) {
        source = &mouse->sources[i];
        if (source->mouseID == mouseID) {
            match = source;
            break;
        }
    }

    if (!down && (!match || !(match->buttonstate & SDL_BUTTON_MASK(button)))) {
        /* This might be a button release from a transition between mouse messages and raw input.
         * See if there's another mouse source that already has that button down and use that.
         */
        for (i = 0; i < mouse->num_sources; ++i) {
            source = &mouse->sources[i];
            if ((source->buttonstate & SDL_BUTTON_MASK(button))) {
                match = source;
                break;
            }
        }
    }
    if (match) {
        return match;
    }

    sources = (SDL_MouseInputSource *)SDL_realloc(mouse->sources, (mouse->num_sources + 1) * sizeof(*mouse->sources));
    if (sources) {
        mouse->sources = sources;
        ++mouse->num_sources;
        source = &sources[mouse->num_sources - 1];
        SDL_zerop(source);
        source->mouseID = mouseID;
        return source;
    }
    return NULL;
}

static SDL_MouseClickState *GetMouseClickState(SDL_MouseInputSource *source, Uint8 button)
{
    if (button >= source->num_clickstates) {
        int i, count = button + 1;
        SDL_MouseClickState *clickstate = (SDL_MouseClickState *)SDL_realloc(source->clickstate, count * sizeof(*source->clickstate));
        if (!clickstate) {
            return NULL;
        }
        source->clickstate = clickstate;

        for (i = source->num_clickstates; i < count; ++i) {
            SDL_zero(source->clickstate[i]);
        }
        source->num_clickstates = count;
    }
    return &source->clickstate[button];
}

static void SDL_PrivateSendMouseButton(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, Uint8 button, bool down, int clicks)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_EventType type;
    Uint32 buttonstate;
    SDL_MouseInputSource *source;

    source = GetMouseInputSource(mouse, mouseID, down, button);
    if (!source) {
        return;
    }
    buttonstate = source->buttonstate;

    // SDL_HINT_MOUSE_TOUCH_EVENTS: controlling whether mouse events should generate synthetic touch events
    if (mouse->mouse_touch_events) {
        if (mouseID != SDL_TOUCH_MOUSEID && mouseID != SDL_PEN_MOUSEID && button == SDL_BUTTON_LEFT) {
            if (down) {
                track_mouse_down = true;
            } else {
                track_mouse_down = false;
            }
            if (window) {
                type = track_mouse_down ? SDL_EVENT_FINGER_DOWN : SDL_EVENT_FINGER_UP;
                float normalized_x = mouse->x / (float)window->w;
                float normalized_y = mouse->y / (float)window->h;
                SDL_SendTouch(timestamp, SDL_MOUSE_TOUCHID, SDL_BUTTON_LEFT, window, type, normalized_x, normalized_y, 1.0f);
            }
        }
    }

    // SDL_HINT_TOUCH_MOUSE_EVENTS: if not set, discard synthetic mouse events coming from platform layer
    if (mouse->touch_mouse_events == 0) {
        if (mouseID == SDL_TOUCH_MOUSEID) {
            return;
        }
    }

    // Figure out which event to perform
    if (down) {
        type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        buttonstate |= SDL_BUTTON_MASK(button);
    } else {
        type = SDL_EVENT_MOUSE_BUTTON_UP;
        buttonstate &= ~SDL_BUTTON_MASK(button);
    }

    // We do this after calculating buttonstate so button presses gain focus
    if (window && down) {
        SDL_UpdateMouseFocus(window, mouse->x, mouse->y, buttonstate, true);
    }

    if (buttonstate == source->buttonstate) {
        // Ignore this event, no state change
        return;
    }
    source->buttonstate = buttonstate;

    if (clicks < 0) {
        SDL_MouseClickState *clickstate = GetMouseClickState(source, button);
        if (clickstate) {
            if (down) {
                Uint64 now = SDL_GetTicks();

                if (now >= (clickstate->last_timestamp + mouse->double_click_time) ||
                    SDL_fabs(mouse->click_motion_x - clickstate->click_motion_x) > mouse->double_click_radius ||
                    SDL_fabs(mouse->click_motion_y - clickstate->click_motion_y) > mouse->double_click_radius) {
                    clickstate->click_count = 0;
                }
                clickstate->last_timestamp = now;
                clickstate->click_motion_x = mouse->click_motion_x;
                clickstate->click_motion_y = mouse->click_motion_y;
                if (clickstate->click_count < 255) {
                    ++clickstate->click_count;
                }
            }
            clicks = clickstate->click_count;
        } else {
            clicks = 1;
        }
    }

    // Post the event, if desired
    if (SDL_EventEnabled(type)) {
        if ((!mouse->relative_mode || mouse->warp_emulation_active) && mouseID != SDL_TOUCH_MOUSEID && mouseID != SDL_PEN_MOUSEID) {
            // We're not in relative mode, so all mouse events are global mouse events
            mouseID = SDL_GLOBAL_MOUSE_ID;
        } else {
            mouseID = source->mouseID;
        }

        SDL_Event event;
        event.type = type;
        event.common.timestamp = timestamp;
        event.button.windowID = mouse->focus ? mouse->focus->id : 0;
        event.button.which = mouseID;
        event.button.down = down;
        event.button.button = button;
        event.button.clicks = (Uint8)SDL_min(clicks, 255);
        event.button.x = mouse->x;
        event.button.y = mouse->y;
        SDL_PushEvent(&event);
    }

    // We do this after dispatching event so button releases can lose focus
    if (window && !down) {
        SDL_UpdateMouseFocus(window, mouse->x, mouse->y, buttonstate, true);
    }

    // Automatically capture the mouse while buttons are pressed
    if (mouse->auto_capture) {
        SDL_UpdateMouseCapture(false);
    }
}

void SDL_SendMouseButtonClicks(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, Uint8 button, bool down, int clicks)
{
    clicks = SDL_max(clicks, 0);
    SDL_PrivateSendMouseButton(timestamp, window, mouseID, button, down, clicks);
}

void SDL_SendMouseButton(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, Uint8 button, bool down)
{
    SDL_PrivateSendMouseButton(timestamp, window, mouseID, button, down, -1);
}

void SDL_SendMouseWheel(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, float x, float y, SDL_MouseWheelDirection direction)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (window) {
        SDL_SetMouseFocus(window);
    }

    // Accumulate fractional wheel motion if integer mode is enabled
    if (mouse->integer_mode_flags & 2) {
        mouse->integer_mode_residual_scroll_x = SDL_modff(mouse->integer_mode_residual_scroll_x + x, &x);
        mouse->integer_mode_residual_scroll_y = SDL_modff(mouse->integer_mode_residual_scroll_y + y, &y);
    }

    if (x == 0.0f && y == 0.0f) {
        return;
    }

    // Post the event, if desired
    if (SDL_EventEnabled(SDL_EVENT_MOUSE_WHEEL)) {
        if (!mouse->relative_mode || mouse->warp_emulation_active) {
            // We're not in relative mode, so all mouse events are global mouse events
            mouseID = SDL_GLOBAL_MOUSE_ID;
        }

        SDL_Event event;
        event.type = SDL_EVENT_MOUSE_WHEEL;
        event.common.timestamp = timestamp;
        event.wheel.windowID = mouse->focus ? mouse->focus->id : 0;
        event.wheel.which = mouseID;
        event.wheel.x = x;
        event.wheel.y = y;
        event.wheel.direction = direction;
        event.wheel.mouse_x = mouse->x;
        event.wheel.mouse_y = mouse->y;
        SDL_PushEvent(&event);
    }
}

void SDL_QuitMouse(void)
{
    SDL_Cursor *cursor, *next;
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->added_mouse_touch_device) {
        SDL_DelTouch(SDL_MOUSE_TOUCHID);
        mouse->added_mouse_touch_device = false;
    }

    if (mouse->added_pen_touch_device) {
        SDL_DelTouch(SDL_PEN_TOUCHID);
        mouse->added_pen_touch_device = false;
    }

    if (mouse->CaptureMouse) {
        SDL_CaptureMouse(false);
        SDL_UpdateMouseCapture(true);
    }
    SDL_SetRelativeMouseMode(false);
    SDL_ShowCursor();

    if (mouse->def_cursor) {
        SDL_SetDefaultCursor(NULL);
    }

    cursor = mouse->cursors;
    while (cursor) {
        next = cursor->next;
        SDL_DestroyCursor(cursor);
        cursor = next;
    }
    mouse->cursors = NULL;
    mouse->cur_cursor = NULL;

    if (mouse->sources) {
        for (int i = 0; i < mouse->num_sources; ++i) {
            SDL_MouseInputSource *source = &mouse->sources[i];
            SDL_free(source->clickstate);
        }
        SDL_free(mouse->sources);
        mouse->sources = NULL;
    }
    mouse->num_sources = 0;

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_DOUBLE_CLICK_TIME,
                        SDL_MouseDoubleClickTimeChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_DOUBLE_CLICK_RADIUS,
                        SDL_MouseDoubleClickRadiusChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_NORMAL_SPEED_SCALE,
                        SDL_MouseNormalSpeedScaleChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_RELATIVE_SPEED_SCALE,
                        SDL_MouseRelativeSpeedScaleChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_RELATIVE_SYSTEM_SCALE,
                        SDL_MouseRelativeSystemScaleChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_RELATIVE_MODE_CENTER,
                        SDL_MouseRelativeModeCenterChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_EMULATE_WARP_WITH_RELATIVE,
                        SDL_MouseWarpEmulationChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_TOUCH_MOUSE_EVENTS,
                        SDL_TouchMouseEventsChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_TOUCH_EVENTS,
                        SDL_MouseTouchEventsChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_PEN_MOUSE_EVENTS,
                        SDL_PenMouseEventsChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_PEN_TOUCH_EVENTS,
                        SDL_PenTouchEventsChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_AUTO_CAPTURE,
                        SDL_MouseAutoCaptureChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_RELATIVE_WARP_MOTION,
                        SDL_MouseRelativeWarpMotionChanged, mouse);

    SDL_RemoveHintCallback(SDL_HINT_MOUSE_RELATIVE_CURSOR_VISIBLE,
                        SDL_MouseRelativeCursorVisibleChanged, mouse);

    SDL_RemoveHintCallback("SDL_MOUSE_INTEGER_MODE",
                        SDL_MouseIntegerModeChanged, mouse);

    for (int i = SDL_mouse_count; i--; ) {
        SDL_RemoveMouse(SDL_mice[i].instance_id, false);
    }
    SDL_free(SDL_mice);
    SDL_mice = NULL;
}

bool SDL_SetRelativeMouseTransform(SDL_MouseMotionTransformCallback transform, void *userdata)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    if (mouse->relative_mode) {
        return SDL_SetError("Can't set mouse transform while relative mode is active");
    }
    mouse->InputTransform = transform;
    mouse->input_transform_data = userdata;
    return true;
}

SDL_MouseButtonFlags SDL_GetMouseState(float *x, float *y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (x) {
        *x = mouse->x;
    }
    if (y) {
        *y = mouse->y;
    }
    return SDL_GetMouseButtonState(mouse, SDL_GLOBAL_MOUSE_ID, true);
}

SDL_MouseButtonFlags SDL_GetRelativeMouseState(float *x, float *y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (x) {
        *x = mouse->x_accu;
    }
    if (y) {
        *y = mouse->y_accu;
    }
    mouse->x_accu = 0.0f;
    mouse->y_accu = 0.0f;
    return SDL_GetMouseButtonState(mouse, SDL_GLOBAL_MOUSE_ID, true);
}

SDL_MouseButtonFlags SDL_GetGlobalMouseState(float *x, float *y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->GetGlobalMouseState) {
        float tmpx, tmpy;

        // make sure these are never NULL for the backend implementations...
        if (!x) {
            x = &tmpx;
        }
        if (!y) {
            y = &tmpy;
        }

        *x = *y = 0.0f;

        return mouse->GetGlobalMouseState(x, y);
    } else {
        return SDL_GetMouseState(x, y);
    }
}

void SDL_PerformWarpMouseInWindow(SDL_Window *window, float x, float y, bool ignore_relative_mode)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!window) {
        window = mouse->focus;
    }

    if (!window) {
        return;
    }

    if ((window->flags & SDL_WINDOW_MINIMIZED) == SDL_WINDOW_MINIMIZED) {
        return;
    }

    // Ignore the previous position when we warp
    mouse->last_x = x;
    mouse->last_y = y;
    mouse->has_position = false;

    if (mouse->relative_mode && !ignore_relative_mode) {
        /* 2.0.22 made warping in relative mode actually functional, which
         * surprised many applications that weren't expecting the additional
         * mouse motion.
         *
         * So for now, warping in relative mode adjusts the absolution position
         * but doesn't generate motion events, unless SDL_HINT_MOUSE_RELATIVE_WARP_MOTION is set.
         */
        if (!mouse->relative_mode_warp_motion) {
            mouse->x = x;
            mouse->y = y;
            mouse->has_position = true;
            return;
        }
    }

    if (mouse->WarpMouse && !mouse->relative_mode) {
        mouse->WarpMouse(window, x, y);
    } else {
        SDL_PrivateSendMouseMotion(0, window, SDL_GLOBAL_MOUSE_ID, false, x, y);
    }
}

void SDL_DisableMouseWarpEmulation(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->warp_emulation_active) {
        SDL_SetRelativeMouseMode(false);
    }

    mouse->warp_emulation_prohibited = true;
}

static void SDL_MaybeEnableWarpEmulation(SDL_Window *window, float x, float y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!mouse->warp_emulation_prohibited && mouse->warp_emulation_hint && !mouse->cursor_visible && !mouse->warp_emulation_active) {
        if (!window) {
            window = mouse->focus;
        }

        if (window) {
            const float cx = window->w / 2.f;
            const float cy = window->h / 2.f;
            if (x >= SDL_floorf(cx) && x <= SDL_ceilf(cx) &&
                y >= SDL_floorf(cy) && y <= SDL_ceilf(cy)) {

                // Require two consecutive warps to the center within a certain timespan to enter warp emulation mode.
                const Uint64 now = SDL_GetTicksNS();
                if (now - mouse->last_center_warp_time_ns < WARP_EMULATION_THRESHOLD_NS) {
                    if (SDL_SetRelativeMouseMode(true)) {
                        mouse->warp_emulation_active = true;
                    }
                }

                mouse->last_center_warp_time_ns = now;
                return;
            }
        }

        mouse->last_center_warp_time_ns = 0;
    }
}

void SDL_WarpMouseInWindow(SDL_Window *window, float x, float y)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_MaybeEnableWarpEmulation(window, x, y);

    SDL_PerformWarpMouseInWindow(window, x, y, mouse->warp_emulation_active);
}

bool SDL_WarpMouseGlobal(float x, float y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->WarpMouseGlobal) {
        return mouse->WarpMouseGlobal(x, y);
    }

    return SDL_Unsupported();
}

bool SDL_SetRelativeMouseMode(bool enabled)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Window *focusWindow = SDL_GetKeyboardFocus();

    if (!enabled) {
        // If warps were being emulated, reset the flag.
        mouse->warp_emulation_active = false;
    }

    if (enabled == mouse->relative_mode) {
        return true;
    }

    // Set the relative mode
    if (!mouse->SetRelativeMouseMode || !mouse->SetRelativeMouseMode(enabled)) {
        if (enabled) {
            return SDL_SetError("No relative mode implementation available");
        }
    }
    mouse->relative_mode = enabled;

    if (enabled) {
        // Update cursor visibility before we potentially warp the mouse
        SDL_RedrawCursor();
    }

    if (enabled && focusWindow) {
        SDL_SetMouseFocus(focusWindow);
    }

    if (focusWindow) {
        SDL_UpdateWindowGrab(focusWindow);

        // Put the cursor back to where the application expects it
        if (!enabled) {
            SDL_PerformWarpMouseInWindow(focusWindow, mouse->x, mouse->y, true);
        }

        SDL_UpdateMouseCapture(false);
    }

    if (!enabled) {
        // Update cursor visibility after we restore the mouse position
        SDL_RedrawCursor();
    }

    // Flush pending mouse motion - ideally we would pump events, but that's not always safe
    SDL_FlushEvent(SDL_EVENT_MOUSE_MOTION);

    return true;
}

bool SDL_GetRelativeMouseMode(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    return mouse->relative_mode;
}

void SDL_UpdateRelativeMouseMode(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Window *focus = SDL_GetKeyboardFocus();
    bool relative_mode = (focus && (focus->flags & SDL_WINDOW_MOUSE_RELATIVE_MODE));

    if (relative_mode != mouse->relative_mode) {
        SDL_SetRelativeMouseMode(relative_mode);
    }
}

bool SDL_UpdateMouseCapture(bool force_release)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Window *capture_window = NULL;

    if (!mouse->CaptureMouse) {
        return true;
    }

    if (!force_release) {
        if (SDL_GetMessageBoxCount() == 0 &&
            (mouse->capture_desired || (mouse->auto_capture && SDL_GetMouseButtonState(mouse, SDL_GLOBAL_MOUSE_ID, false) != 0))) {
            if (!mouse->relative_mode) {
                capture_window = mouse->focus;
            }
        }
    }

    if (capture_window != mouse->capture_window) {
        /* We can get here recursively on Windows, so make sure we complete
         * all of the window state operations before we change the capture state
         * (e.g. https://github.com/libsdl-org/SDL/pull/5608)
         */
        SDL_Window *previous_capture = mouse->capture_window;

        if (previous_capture) {
            previous_capture->flags &= ~SDL_WINDOW_MOUSE_CAPTURE;
        }

        if (capture_window) {
            capture_window->flags |= SDL_WINDOW_MOUSE_CAPTURE;
        }

        mouse->capture_window = capture_window;

        if (!mouse->CaptureMouse(capture_window)) {
            // CaptureMouse() will have set an error, just restore the state
            if (previous_capture) {
                previous_capture->flags |= SDL_WINDOW_MOUSE_CAPTURE;
            }
            if (capture_window) {
                capture_window->flags &= ~SDL_WINDOW_MOUSE_CAPTURE;
            }
            mouse->capture_window = previous_capture;

            return false;
        }
    }
    return true;
}

bool SDL_CaptureMouse(bool enabled)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!mouse->CaptureMouse) {
        return SDL_Unsupported();
    }

#if defined(SDL_PLATFORM_WIN32) || defined(SDL_PLATFORM_WINGDK)
    /* Windows mouse capture is tied to the current thread, and must be called
     * from the thread that created the window being captured. Since we update
     * the mouse capture state from the event processing, any application state
     * changes must be processed on that thread as well.
     */
    if (!SDL_OnVideoThread()) {
        return SDL_SetError("SDL_CaptureMouse() must be called on the main thread");
    }
#endif // defined(SDL_PLATFORM_WIN32) || defined(SDL_PLATFORM_WINGDK)

    if (enabled && SDL_GetKeyboardFocus() == NULL) {
        return SDL_SetError("No window has focus");
    }
    mouse->capture_desired = enabled;

    return SDL_UpdateMouseCapture(false);
}

SDL_Cursor *SDL_CreateCursor(const Uint8 *data, const Uint8 *mask, int w, int h, int hot_x, int hot_y)
{
    SDL_Surface *surface;
    SDL_Cursor *cursor;
    int x, y;
    Uint32 *pixels;
    Uint8 datab = 0, maskb = 0;
    const Uint32 black = 0xFF000000;
    const Uint32 white = 0xFFFFFFFF;
    const Uint32 transparent = 0x00000000;
#if defined(SDL_PLATFORM_WIN32)
    // Only Windows backend supports inverted pixels in mono cursors.
    const Uint32 inverted = 0x00FFFFFF;
#else
    const Uint32 inverted = 0xFF000000;
#endif // defined(SDL_PLATFORM_WIN32)

    // Make sure the width is a multiple of 8
    w = ((w + 7) & ~7);

    // Create the surface from a bitmap
    surface = SDL_CreateSurface(w, h, SDL_PIXELFORMAT_ARGB8888);
    if (!surface) {
        return NULL;
    }
    for (y = 0; y < h; ++y) {
        pixels = (Uint32 *)((Uint8 *)surface->pixels + y * surface->pitch);
        for (x = 0; x < w; ++x) {
            if ((x % 8) == 0) {
                datab = *data++;
                maskb = *mask++;
            }
            if (maskb & 0x80) {
                *pixels++ = (datab & 0x80) ? black : white;
            } else {
                *pixels++ = (datab & 0x80) ? inverted : transparent;
            }
            datab <<= 1;
            maskb <<= 1;
        }
    }

    cursor = SDL_CreateColorCursor(surface, hot_x, hot_y);

    SDL_DestroySurface(surface);

    return cursor;
}

SDL_Cursor *SDL_CreateColorCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Surface *temp = NULL;
    SDL_Cursor *cursor;

    if (!surface) {
        SDL_InvalidParamError("surface");
        return NULL;
    }

    // Allow specifying the hot spot via properties on the surface
    SDL_PropertiesID props = SDL_GetSurfaceProperties(surface);
    hot_x = (int)SDL_GetNumberProperty(props, SDL_PROP_SURFACE_HOTSPOT_X_NUMBER, hot_x);
    hot_y = (int)SDL_GetNumberProperty(props, SDL_PROP_SURFACE_HOTSPOT_Y_NUMBER, hot_y);

    // Sanity check the hot spot
    if ((hot_x < 0) || (hot_y < 0) ||
        (hot_x >= surface->w) || (hot_y >= surface->h)) {
        SDL_SetError("Cursor hot spot doesn't lie within cursor");
        return NULL;
    }

    if (surface->format != SDL_PIXELFORMAT_ARGB8888) {
        temp = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_ARGB8888);
        if (!temp) {
            return NULL;
        }
        surface = temp;
    }

    if (mouse->CreateCursor) {
        cursor = mouse->CreateCursor(surface, hot_x, hot_y);
    } else {
        cursor = (SDL_Cursor *)SDL_calloc(1, sizeof(*cursor));
    }
    if (cursor) {
        cursor->next = mouse->cursors;
        mouse->cursors = cursor;
    }

    SDL_DestroySurface(temp);

    return cursor;
}

SDL_Cursor *SDL_CreateSystemCursor(SDL_SystemCursor id)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Cursor *cursor;

    if (!mouse->CreateSystemCursor) {
        SDL_SetError("CreateSystemCursor is not currently supported");
        return NULL;
    }

    cursor = mouse->CreateSystemCursor(id);
    if (cursor) {
        cursor->next = mouse->cursors;
        mouse->cursors = cursor;
    }

    return cursor;
}

void SDL_RedrawCursor(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Cursor *cursor;

    if (mouse->focus) {
        cursor = mouse->cur_cursor;
    } else {
        cursor = mouse->def_cursor;
    }

    if (mouse->focus && (!mouse->cursor_visible || (mouse->relative_mode && mouse->relative_mode_hide_cursor))) {
        cursor = NULL;
    }

    if (mouse->ShowCursor) {
        mouse->ShowCursor(cursor);
    }
}

/* SDL_SetCursor(NULL) can be used to force the cursor redraw,
   if this is desired for any reason.  This is used when setting
   the video mode and when the SDL window gains the mouse focus.
 */
bool SDL_SetCursor(SDL_Cursor *cursor)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    // already on this cursor, no further action required
    if (cursor == mouse->cur_cursor) {
        return true;
    }

    // Set the new cursor
    if (cursor) {
        // Make sure the cursor is still valid for this mouse
        if (cursor != mouse->def_cursor) {
            SDL_Cursor *found;
            for (found = mouse->cursors; found; found = found->next) {
                if (found == cursor) {
                    break;
                }
            }
            if (!found) {
                return SDL_SetError("Cursor not associated with the current mouse");
            }
        }
        mouse->cur_cursor = cursor;
    }

    SDL_RedrawCursor();

    return true;
}

SDL_Cursor *SDL_GetCursor(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!mouse) {
        return NULL;
    }
    return mouse->cur_cursor;
}

SDL_Cursor *SDL_GetDefaultCursor(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!mouse) {
        return NULL;
    }
    return mouse->def_cursor;
}

void SDL_DestroyCursor(SDL_Cursor *cursor)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_Cursor *curr, *prev;

    if (!cursor) {
        return;
    }

    if (cursor == mouse->def_cursor) {
        return;
    }
    if (cursor == mouse->cur_cursor) {
        SDL_SetCursor(mouse->def_cursor);
    }

    for (prev = NULL, curr = mouse->cursors; curr;
         prev = curr, curr = curr->next) {
        if (curr == cursor) {
            if (prev) {
                prev->next = curr->next;
            } else {
                mouse->cursors = curr->next;
            }

            if (mouse->FreeCursor && curr->internal) {
                mouse->FreeCursor(curr);
            } else {
                SDL_free(curr);
            }
            return;
        }
    }
}

bool SDL_ShowCursor(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->warp_emulation_active) {
        SDL_SetRelativeMouseMode(false);
        mouse->warp_emulation_active = false;
    }

    if (!mouse->cursor_visible) {
        mouse->cursor_visible = true;
        SDL_RedrawCursor();
    }
    return true;
}

bool SDL_HideCursor(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (mouse->cursor_visible) {
        mouse->cursor_visible = false;
        SDL_RedrawCursor();
    }
    return true;
}

bool SDL_CursorVisible(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    return mouse->cursor_visible;
}
