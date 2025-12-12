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

// General event handling code for SDL

#include "SDL_events_c.h"
#include "SDL_eventwatch_c.h"
#include "../SDL_hints_c.h"
#include "../timer/SDL_timer_c.h"
#ifndef SDL_JOYSTICK_DISABLED
#include "../joystick/SDL_joystick_c.h"
#endif
#ifndef SDL_SENSOR_DISABLED
#include "../sensor/SDL_sensor_c.h"
#endif
//#include "../video/SDL_sysvideo.h"

#ifdef SDL_PLATFORM_ANDROID
#include "../core/android/SDL_android.h"
#include "../video/android/SDL_androidevents.h"
#endif

// An arbitrary limit so we don't have unbounded growth
#define SDL_MAX_QUEUED_EVENTS 65535

// Determines how often we pump events if joystick or sensor subsystems are active
#define ENUMERATION_POLL_INTERVAL_NS (3 * SDL_NS_PER_SECOND)

// Determines how often to pump events if joysticks or sensors are actively being read
#define EVENT_POLL_INTERVAL_NS SDL_MS_TO_NS(1)

// Make sure the type in the SDL_Event aligns properly across the union
SDL_COMPILE_TIME_ASSERT(SDL_Event_type, sizeof(Uint32) == sizeof(SDL_EventType));

#define SDL2_SYSWMEVENT 0x201

#ifdef SDL_VIDEO_DRIVER_WINDOWS
#include "../core/windows/SDL_windows.h"
#endif

#ifdef SDL_VIDEO_DRIVER_X11
#include <X11/Xlib.h>
#endif

typedef struct SDL2_version
{
    Uint8 major;
    Uint8 minor;
    Uint8 patch;
} SDL2_version;

typedef enum
{
  SDL2_SYSWM_UNKNOWN
} SDL2_SYSWM_TYPE;

typedef struct SDL2_SysWMmsg
{
    SDL2_version version;
    SDL2_SYSWM_TYPE subsystem;
    union
    {
#ifdef SDL_VIDEO_DRIVER_WINDOWS
        struct {
            HWND hwnd;                  /**< The window for the message */
            UINT msg;                   /**< The type of message */
            WPARAM wParam;              /**< WORD message parameter */
            LPARAM lParam;              /**< LONG message parameter */
        } win;
#endif
#ifdef SDL_VIDEO_DRIVER_X11
        struct {
            XEvent event;
        } x11;
#endif
        /* Can't have an empty union */
        int dummy;
    } msg;
} SDL2_SysWMmsg;

static SDL_EventWatchList SDL_event_watchers;
static SDL_AtomicInt SDL_sentinel_pending;
static Uint32 SDL_last_event_id = 0;

typedef struct
{
    Uint32 bits[8];
} SDL_DisabledEventBlock;

static SDL_DisabledEventBlock *SDL_disabled_events[256];
static SDL_AtomicInt SDL_userevents;

typedef struct SDL_TemporaryMemory
{
    void *memory;
    struct SDL_TemporaryMemory *prev;
    struct SDL_TemporaryMemory *next;
} SDL_TemporaryMemory;

typedef struct SDL_TemporaryMemoryState
{
    SDL_TemporaryMemory *head;
    SDL_TemporaryMemory *tail;
} SDL_TemporaryMemoryState;

static SDL_TLSID SDL_temporary_memory;

typedef struct SDL_EventEntry
{
    SDL_Event event;
    SDL_TemporaryMemory *memory;
    struct SDL_EventEntry *prev;
    struct SDL_EventEntry *next;
} SDL_EventEntry;

static struct
{
    SDL_Mutex *lock;
    bool active;
    SDL_AtomicInt count;
    int max_events_seen;
    SDL_EventEntry *head;
    SDL_EventEntry *tail;
    SDL_EventEntry *free;
} SDL_EventQ = { NULL, false, { 0 }, 0, NULL, NULL, NULL };


static void SDL_CleanupTemporaryMemory(void *data)
{
    SDL_TemporaryMemoryState *state = (SDL_TemporaryMemoryState *)data;

    SDL_FreeTemporaryMemory();
    SDL_free(state);
}

static SDL_TemporaryMemoryState *SDL_GetTemporaryMemoryState(bool create)
{
    SDL_TemporaryMemoryState *state;

    state = (SDL_TemporaryMemoryState *)SDL_GetTLS(&SDL_temporary_memory);
    if (!state) {
        if (!create) {
            return NULL;
        }

        state = (SDL_TemporaryMemoryState *)SDL_calloc(1, sizeof(*state));
        if (!state) {
            return NULL;
        }

        if (!SDL_SetTLS(&SDL_temporary_memory, state, SDL_CleanupTemporaryMemory)) {
            SDL_free(state);
            return NULL;
        }
    }
    return state;
}

static SDL_TemporaryMemory *SDL_GetTemporaryMemoryEntry(SDL_TemporaryMemoryState *state, const void *mem)
{
    SDL_TemporaryMemory *entry;

    // Start from the end, it's likely to have been recently allocated
    for (entry = state->tail; entry; entry = entry->prev) {
        if (mem == entry->memory) {
            return entry;
        }
    }
    return NULL;
}

static void SDL_LinkTemporaryMemoryEntry(SDL_TemporaryMemoryState *state, SDL_TemporaryMemory *entry)
{
    entry->prev = state->tail;
    entry->next = NULL;

    if (state->tail) {
        state->tail->next = entry;
    } else {
        state->head = entry;
    }
    state->tail = entry;
}

static void SDL_UnlinkTemporaryMemoryEntry(SDL_TemporaryMemoryState *state, SDL_TemporaryMemory *entry)
{
    if (state->head == entry) {
        state->head = entry->next;
    }
    if (state->tail == entry) {
        state->tail = entry->prev;
    }

    if (entry->prev) {
        entry->prev->next = entry->next;
    }
    if (entry->next) {
        entry->next->prev = entry->prev;
    }

    entry->prev = NULL;
    entry->next = NULL;
}

static void SDL_FreeTemporaryMemoryEntry(SDL_TemporaryMemoryState *state, SDL_TemporaryMemory *entry, bool free_data)
{
    if (free_data) {
        SDL_free(entry->memory);
    }
    SDL_free(entry);
}

static void SDL_LinkTemporaryMemoryToEvent(SDL_EventEntry *event, const void *mem)
{
    SDL_TemporaryMemoryState *state;
    SDL_TemporaryMemory *entry;

    state = SDL_GetTemporaryMemoryState(false);
    if (!state) {
        return;
    }

    entry = SDL_GetTemporaryMemoryEntry(state, mem);
    if (entry) {
        SDL_UnlinkTemporaryMemoryEntry(state, entry);
        entry->next = event->memory;
        event->memory = entry;
    }
}

static void SDL_TransferSysWMMemoryToEvent(SDL_EventEntry *event)
{
    SDL2_SysWMmsg **wmmsg = (SDL2_SysWMmsg **)((&event->event.common)+1);
    SDL2_SysWMmsg *mem = SDL_AllocateTemporaryMemory(sizeof(*mem));
    if (mem) {
        SDL_copyp(mem, *wmmsg);
        *wmmsg = mem;
        SDL_LinkTemporaryMemoryToEvent(event, mem);
    }
}

// Transfer the event memory from the thread-local event memory list to the event
static void SDL_TransferTemporaryMemoryToEvent(SDL_EventEntry *event)
{
    switch (event->event.type) {
    case SDL_EVENT_TEXT_EDITING:
        SDL_LinkTemporaryMemoryToEvent(event, event->event.edit.text);
        break;
    case SDL_EVENT_TEXT_EDITING_CANDIDATES:
        SDL_LinkTemporaryMemoryToEvent(event, event->event.edit_candidates.candidates);
        break;
    case SDL_EVENT_TEXT_INPUT:
        SDL_LinkTemporaryMemoryToEvent(event, event->event.text.text);
        break;
    case SDL_EVENT_DROP_BEGIN:
    case SDL_EVENT_DROP_FILE:
    case SDL_EVENT_DROP_TEXT:
    case SDL_EVENT_DROP_COMPLETE:
    case SDL_EVENT_DROP_POSITION:
        SDL_LinkTemporaryMemoryToEvent(event, event->event.drop.source);
        SDL_LinkTemporaryMemoryToEvent(event, event->event.drop.data);
        break;
    case SDL_EVENT_CLIPBOARD_UPDATE:
        SDL_LinkTemporaryMemoryToEvent(event, event->event.clipboard.mime_types);
        break;
    case SDL2_SYSWMEVENT:
        // We need to copy the stack pointer into temporary memory
        SDL_TransferSysWMMemoryToEvent(event);
        break;
    default:
        break;
    }
}

// Transfer the event memory from the event to the thread-local event memory list
static void SDL_TransferTemporaryMemoryFromEvent(SDL_EventEntry *event)
{
    SDL_TemporaryMemoryState *state;
    SDL_TemporaryMemory *entry, *next;

    if (!event->memory) {
        return;
    }

    state = SDL_GetTemporaryMemoryState(true);
    if (!state) {
        return;  // this is now a leak, but you probably have bigger problems if malloc failed.
    }

    for (entry = event->memory; entry; entry = next) {
        next = entry->next;
        SDL_LinkTemporaryMemoryEntry(state, entry);
    }
    event->memory = NULL;
}

static void *SDL_FreeLater(void *memory)
{
    SDL_TemporaryMemoryState *state;

    if (memory == NULL) {
        return NULL;
    }

    // Make sure we're not adding this to the list twice
    //SDL_assert(!SDL_ClaimTemporaryMemory(memory));

    state = SDL_GetTemporaryMemoryState(true);
    if (!state) {
        return memory;  // this is now a leak, but you probably have bigger problems if malloc failed.
    }

    SDL_TemporaryMemory *entry = (SDL_TemporaryMemory *)SDL_malloc(sizeof(*entry));
    if (!entry) {
        return memory;  // this is now a leak, but you probably have bigger problems if malloc failed. We could probably pool up and reuse entries, though.
    }

    entry->memory = memory;

    SDL_LinkTemporaryMemoryEntry(state, entry);

    return memory;
}

void *SDL_AllocateTemporaryMemory(size_t size)
{
    return SDL_FreeLater(SDL_malloc(size));
}

const char *SDL_CreateTemporaryString(const char *string)
{
    if (string) {
        return (const char *)SDL_FreeLater(SDL_strdup(string));
    }
    return NULL;
}

void *SDL_ClaimTemporaryMemory(const void *mem)
{
    SDL_TemporaryMemoryState *state;

    state = SDL_GetTemporaryMemoryState(false);
    if (state && mem) {
        SDL_TemporaryMemory *entry = SDL_GetTemporaryMemoryEntry(state, mem);
        if (entry) {
            SDL_UnlinkTemporaryMemoryEntry(state, entry);
            SDL_FreeTemporaryMemoryEntry(state, entry, false);
            return (void *)mem;
        }
    }
    return NULL;
}

void SDL_FreeTemporaryMemory(void)
{
    SDL_TemporaryMemoryState *state;

    state = SDL_GetTemporaryMemoryState(false);
    if (!state) {
        return;
    }

    while (state->head) {
        SDL_TemporaryMemory *entry = state->head;

        SDL_UnlinkTemporaryMemoryEntry(state, entry);
        SDL_FreeTemporaryMemoryEntry(state, entry, true);
    }
}

#ifndef SDL_JOYSTICK_DISABLED

static bool SDL_update_joysticks = true;

static void SDLCALL SDL_AutoUpdateJoysticksChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_update_joysticks = SDL_GetStringBoolean(hint, true);
}

#endif // !SDL_JOYSTICK_DISABLED

#ifndef SDL_SENSOR_DISABLED

static bool SDL_update_sensors = true;

static void SDLCALL SDL_AutoUpdateSensorsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_update_sensors = SDL_GetStringBoolean(hint, true);
}

#endif // !SDL_SENSOR_DISABLED

static void SDLCALL SDL_PollSentinelChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_SetEventEnabled(SDL_EVENT_POLL_SENTINEL, SDL_GetStringBoolean(hint, true));
}

/**
 * Verbosity of logged events as defined in SDL_HINT_EVENT_LOGGING:
 *  - 0: (default) no logging
 *  - 1: logging of most events
 *  - 2: as above, plus mouse, pen, and finger motion
 */
static int SDL_EventLoggingVerbosity = 0;

static void SDLCALL SDL_EventLoggingChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_EventLoggingVerbosity = (hint && *hint) ? SDL_clamp(SDL_atoi(hint), 0, 3) : 0;
}

static void SDL_LogEvent(const SDL_Event *event)
{
    static const char *pen_axisnames[] = { "PRESSURE", "XTILT", "YTILT", "DISTANCE", "ROTATION", "SLIDER", "TANGENTIAL_PRESSURE" };
    SDL_COMPILE_TIME_ASSERT(pen_axisnames_array_matches, SDL_arraysize(pen_axisnames) == SDL_PEN_AXIS_COUNT);

    char name[64];
    char details[128];

    // sensor/mouse/pen/finger motion are spammy, ignore these if they aren't demanded.
    if ((SDL_EventLoggingVerbosity < 2) &&
        ((event->type == SDL_EVENT_MOUSE_MOTION) ||
         (event->type == SDL_EVENT_FINGER_MOTION) ||
         (event->type == SDL_EVENT_PEN_AXIS) ||
         (event->type == SDL_EVENT_PEN_MOTION) ||
         (event->type == SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION) ||
         (event->type == SDL_EVENT_GAMEPAD_SENSOR_UPDATE) ||
         (event->type == SDL_EVENT_SENSOR_UPDATE))) {
        return;
    }

// this is to make (void)SDL_snprintf() calls cleaner.
#define uint unsigned int

    name[0] = '\0';
    details[0] = '\0';

    // !!! FIXME: This code is kinda ugly, sorry.

    if ((event->type >= SDL_EVENT_USER) && (event->type <= SDL_EVENT_LAST)) {
        char plusstr[16];
        SDL_strlcpy(name, "SDL_EVENT_USER", sizeof(name));
        if (event->type > SDL_EVENT_USER) {
            (void)SDL_snprintf(plusstr, sizeof(plusstr), "+%u", ((uint)event->type) - SDL_EVENT_USER);
        } else {
            plusstr[0] = '\0';
        }
        (void)SDL_snprintf(details, sizeof(details), "%s (timestamp=%u windowid=%u code=%d data1=%p data2=%p)",
                           plusstr, (uint)event->user.timestamp, (uint)event->user.windowID,
                           (int)event->user.code, event->user.data1, event->user.data2);
    }

    switch (event->type) {
#define SDL_EVENT_CASE(x) \
    case x:               \
        SDL_strlcpy(name, #x, sizeof(name));
        SDL_EVENT_CASE(SDL_EVENT_FIRST)
        SDL_strlcpy(details, " (THIS IS PROBABLY A BUG!)", sizeof(details));
        break;
        SDL_EVENT_CASE(SDL_EVENT_QUIT)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u)", (uint)event->quit.timestamp);
        break;
        SDL_EVENT_CASE(SDL_EVENT_TERMINATING)
        break;
        SDL_EVENT_CASE(SDL_EVENT_LOW_MEMORY)
        break;
        SDL_EVENT_CASE(SDL_EVENT_WILL_ENTER_BACKGROUND)
        break;
        SDL_EVENT_CASE(SDL_EVENT_DID_ENTER_BACKGROUND)
        break;
        SDL_EVENT_CASE(SDL_EVENT_WILL_ENTER_FOREGROUND)
        break;
        SDL_EVENT_CASE(SDL_EVENT_DID_ENTER_FOREGROUND)
        break;
        SDL_EVENT_CASE(SDL_EVENT_LOCALE_CHANGED)
        break;
        SDL_EVENT_CASE(SDL_EVENT_SYSTEM_THEME_CHANGED)
        break;
        SDL_EVENT_CASE(SDL_EVENT_KEYMAP_CHANGED)
        break;
        SDL_EVENT_CASE(SDL_EVENT_CLIPBOARD_UPDATE)
        break;

#define SDL_RENDEREVENT_CASE(x)                \
    case x:                                    \
        SDL_strlcpy(name, #x, sizeof(name));   \
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u event=%s windowid=%u)", \
                           (uint)event->display.timestamp, name, (uint)event->render.windowID); \
        break
        SDL_RENDEREVENT_CASE(SDL_EVENT_RENDER_TARGETS_RESET);
        SDL_RENDEREVENT_CASE(SDL_EVENT_RENDER_DEVICE_RESET);
        SDL_RENDEREVENT_CASE(SDL_EVENT_RENDER_DEVICE_LOST);

#define SDL_DISPLAYEVENT_CASE(x)               \
    case x:                                    \
        SDL_strlcpy(name, #x, sizeof(name));   \
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u display=%u event=%s data1=%d, data2=%d)", \
                           (uint)event->display.timestamp, (uint)event->display.displayID, name, (int)event->display.data1, (int)event->display.data2); \
        break
        SDL_DISPLAYEVENT_CASE(SDL_EVENT_DISPLAY_ORIENTATION);
        SDL_DISPLAYEVENT_CASE(SDL_EVENT_DISPLAY_ADDED);
        SDL_DISPLAYEVENT_CASE(SDL_EVENT_DISPLAY_REMOVED);
        SDL_DISPLAYEVENT_CASE(SDL_EVENT_DISPLAY_MOVED);
        SDL_DISPLAYEVENT_CASE(SDL_EVENT_DISPLAY_DESKTOP_MODE_CHANGED);
        SDL_DISPLAYEVENT_CASE(SDL_EVENT_DISPLAY_CURRENT_MODE_CHANGED);
        SDL_DISPLAYEVENT_CASE(SDL_EVENT_DISPLAY_CONTENT_SCALE_CHANGED);
#undef SDL_DISPLAYEVENT_CASE

#define SDL_WINDOWEVENT_CASE(x)                \
    case x:                                    \
        SDL_strlcpy(name, #x, sizeof(name)); \
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u event=%s data1=%d data2=%d)", \
                           (uint)event->window.timestamp, (uint)event->window.windowID, name, (int)event->window.data1, (int)event->window.data2); \
        break
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_SHOWN);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_HIDDEN);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_EXPOSED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_MOVED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_RESIZED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_METAL_VIEW_RESIZED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_SAFE_AREA_CHANGED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_MINIMIZED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_MAXIMIZED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_RESTORED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_MOUSE_ENTER);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_MOUSE_LEAVE);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_FOCUS_GAINED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_FOCUS_LOST);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_CLOSE_REQUESTED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_HIT_TEST);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_ICCPROF_CHANGED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_DISPLAY_CHANGED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_OCCLUDED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_ENTER_FULLSCREEN);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_LEAVE_FULLSCREEN);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_DESTROYED);
        SDL_WINDOWEVENT_CASE(SDL_EVENT_WINDOW_HDR_STATE_CHANGED);
#undef SDL_WINDOWEVENT_CASE

#define PRINT_KEYDEV_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%u)", (uint)event->kdevice.timestamp, (uint)event->kdevice.which)
        SDL_EVENT_CASE(SDL_EVENT_KEYBOARD_ADDED)
        PRINT_KEYDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_KEYBOARD_REMOVED)
        PRINT_KEYDEV_EVENT(event);
        break;
#undef PRINT_KEYDEV_EVENT

#define PRINT_KEY_EVENT(event)                                                                                                              \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u state=%s repeat=%s scancode=%u keycode=%u mod=0x%x)", \
                       (uint)event->key.timestamp, (uint)event->key.windowID, (uint)event->key.which,                                       \
                       event->key.down ? "pressed" : "released",                                                            \
                       event->key.repeat ? "true" : "false",                                                                                \
                       (uint)event->key.scancode,                                                                                           \
                       (uint)event->key.key,                                                                                                \
                       (uint)event->key.mod)
        SDL_EVENT_CASE(SDL_EVENT_KEY_DOWN)
        PRINT_KEY_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_KEY_UP)
        PRINT_KEY_EVENT(event);
        break;
#undef PRINT_KEY_EVENT

        SDL_EVENT_CASE(SDL_EVENT_TEXT_EDITING)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u text='%s' start=%d length=%d)",
                           (uint)event->edit.timestamp, (uint)event->edit.windowID,
                           event->edit.text, (int)event->edit.start, (int)event->edit.length);
        break;

        SDL_EVENT_CASE(SDL_EVENT_TEXT_EDITING_CANDIDATES)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u num_candidates=%d selected_candidate=%d)",
                           (uint)event->edit_candidates.timestamp, (uint)event->edit_candidates.windowID,
                           (int)event->edit_candidates.num_candidates, (int)event->edit_candidates.selected_candidate);
        break;

        SDL_EVENT_CASE(SDL_EVENT_TEXT_INPUT)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u text='%s')", (uint)event->text.timestamp, (uint)event->text.windowID, event->text.text);
        break;

#define PRINT_MOUSEDEV_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%u)", (uint)event->mdevice.timestamp, (uint)event->mdevice.which)
        SDL_EVENT_CASE(SDL_EVENT_MOUSE_ADDED)
        PRINT_MOUSEDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_MOUSE_REMOVED)
        PRINT_MOUSEDEV_EVENT(event);
        break;
#undef PRINT_MOUSEDEV_EVENT

        SDL_EVENT_CASE(SDL_EVENT_MOUSE_MOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u state=%u x=%g y=%g xrel=%g yrel=%g)",
                           (uint)event->motion.timestamp, (uint)event->motion.windowID,
                           (uint)event->motion.which, (uint)event->motion.state,
                           event->motion.x, event->motion.y,
                           event->motion.xrel, event->motion.yrel);
        break;

#define PRINT_MBUTTON_EVENT(event)                                                                                              \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u button=%u state=%s clicks=%u x=%g y=%g)", \
                       (uint)event->button.timestamp, (uint)event->button.windowID,                                             \
                       (uint)event->button.which, (uint)event->button.button,                                                   \
                       event->button.down ? "pressed" : "released",                                                             \
                       (uint)event->button.clicks, event->button.x, event->button.y)
        SDL_EVENT_CASE(SDL_EVENT_MOUSE_BUTTON_DOWN)
        PRINT_MBUTTON_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_MOUSE_BUTTON_UP)
        PRINT_MBUTTON_EVENT(event);
        break;
#undef PRINT_MBUTTON_EVENT

        SDL_EVENT_CASE(SDL_EVENT_MOUSE_WHEEL)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u x=%g y=%g integer_x=%d integer_y=%d direction=%s)",
                           (uint)event->wheel.timestamp, (uint)event->wheel.windowID,
                           (uint)event->wheel.which, event->wheel.x, event->wheel.y,
                           (int)event->wheel.integer_x, (int)event->wheel.integer_y,
                           event->wheel.direction == SDL_MOUSEWHEEL_NORMAL ? "normal" : "flipped");
        break;

        SDL_EVENT_CASE(SDL_EVENT_JOYSTICK_AXIS_MOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d axis=%u value=%d)",
                           (uint)event->jaxis.timestamp, (int)event->jaxis.which,
                           (uint)event->jaxis.axis, (int)event->jaxis.value);
        break;

        SDL_EVENT_CASE(SDL_EVENT_JOYSTICK_BALL_MOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d ball=%u xrel=%d yrel=%d)",
                           (uint)event->jball.timestamp, (int)event->jball.which,
                           (uint)event->jball.ball, (int)event->jball.xrel, (int)event->jball.yrel);
        break;

        SDL_EVENT_CASE(SDL_EVENT_JOYSTICK_HAT_MOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d hat=%u value=%u)",
                           (uint)event->jhat.timestamp, (int)event->jhat.which,
                           (uint)event->jhat.hat, (uint)event->jhat.value);
        break;

#define PRINT_JBUTTON_EVENT(event)                                                              \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d button=%u state=%s)", \
                       (uint)event->jbutton.timestamp, (int)event->jbutton.which,               \
                       (uint)event->jbutton.button, event->jbutton.down ? "pressed" : "released")
        SDL_EVENT_CASE(SDL_EVENT_JOYSTICK_BUTTON_DOWN)
        PRINT_JBUTTON_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_JOYSTICK_BUTTON_UP)
        PRINT_JBUTTON_EVENT(event);
        break;
#undef PRINT_JBUTTON_EVENT

        SDL_EVENT_CASE(SDL_EVENT_JOYSTICK_BATTERY_UPDATED)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d state=%u percent=%d)",
                           (uint)event->jbattery.timestamp, (int)event->jbattery.which,
                           event->jbattery.state, event->jbattery.percent);
        break;

#define PRINT_JOYDEV_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d)", (uint)event->jdevice.timestamp, (int)event->jdevice.which)
        SDL_EVENT_CASE(SDL_EVENT_JOYSTICK_ADDED)
        PRINT_JOYDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_JOYSTICK_REMOVED)
        PRINT_JOYDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_JOYSTICK_UPDATE_COMPLETE)
        PRINT_JOYDEV_EVENT(event);
        break;
#undef PRINT_JOYDEV_EVENT

        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_AXIS_MOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d axis=%u value=%d)",
                           (uint)event->gaxis.timestamp, (int)event->gaxis.which,
                           (uint)event->gaxis.axis, (int)event->gaxis.value);
        break;

#define PRINT_CBUTTON_EVENT(event)                                                              \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d button=%u state=%s)", \
                       (uint)event->gbutton.timestamp, (int)event->gbutton.which,               \
                       (uint)event->gbutton.button, event->gbutton.down ? "pressed" : "released")
        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_BUTTON_DOWN)
        PRINT_CBUTTON_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_BUTTON_UP)
        PRINT_CBUTTON_EVENT(event);
        break;
#undef PRINT_CBUTTON_EVENT

#define PRINT_GAMEPADDEV_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d)", (uint)event->gdevice.timestamp, (int)event->gdevice.which)
        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_ADDED)
        PRINT_GAMEPADDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_REMOVED)
        PRINT_GAMEPADDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_REMAPPED)
        PRINT_GAMEPADDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_UPDATE_COMPLETE)
        PRINT_GAMEPADDEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_STEAM_HANDLE_UPDATED)
        PRINT_GAMEPADDEV_EVENT(event);
        break;
#undef PRINT_GAMEPADDEV_EVENT

#define PRINT_CTOUCHPAD_EVENT(event)                                                                                     \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d touchpad=%d finger=%d x=%f y=%f pressure=%f)", \
                       (uint)event->gtouchpad.timestamp, (int)event->gtouchpad.which,                                    \
                       (int)event->gtouchpad.touchpad, (int)event->gtouchpad.finger,                                     \
                       event->gtouchpad.x, event->gtouchpad.y, event->gtouchpad.pressure)
        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN)
        PRINT_CTOUCHPAD_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_TOUCHPAD_UP)
        PRINT_CTOUCHPAD_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION)
        PRINT_CTOUCHPAD_EVENT(event);
        break;
#undef PRINT_CTOUCHPAD_EVENT

        SDL_EVENT_CASE(SDL_EVENT_GAMEPAD_SENSOR_UPDATE)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d sensor=%d data[0]=%f data[1]=%f data[2]=%f)",
                           (uint)event->gsensor.timestamp, (int)event->gsensor.which, (int)event->gsensor.sensor,
                           event->gsensor.data[0], event->gsensor.data[1], event->gsensor.data[2]);
        break;

#define PRINT_FINGER_EVENT(event)                                                                                                                      \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u touchid=%" SDL_PRIu64 " fingerid=%" SDL_PRIu64 " x=%f y=%f dx=%f dy=%f pressure=%f)", \
                       (uint)event->tfinger.timestamp, event->tfinger.touchID,                                                              \
                       event->tfinger.fingerID, event->tfinger.x, event->tfinger.y,                                                         \
                       event->tfinger.dx, event->tfinger.dy, event->tfinger.pressure)
        SDL_EVENT_CASE(SDL_EVENT_FINGER_DOWN)
        PRINT_FINGER_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_FINGER_UP)
        PRINT_FINGER_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_FINGER_CANCELED)
        PRINT_FINGER_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_FINGER_MOTION)
        PRINT_FINGER_EVENT(event);
        break;
#undef PRINT_FINGER_EVENT

#define PRINT_PTOUCH_EVENT(event)                                                                             \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u pen_state=%u x=%g y=%g eraser=%s state=%s)", \
                       (uint)event->ptouch.timestamp, (uint)event->ptouch.windowID, (uint)event->ptouch.which, (uint)event->ptouch.pen_state, event->ptouch.x, event->ptouch.y, \
                       event->ptouch.eraser ? "yes" : "no", event->ptouch.down ? "down" : "up");
        SDL_EVENT_CASE(SDL_EVENT_PEN_DOWN)
        PRINT_PTOUCH_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_PEN_UP)
        PRINT_PTOUCH_EVENT(event);
        break;
#undef PRINT_PTOUCH_EVENT

#define PRINT_PPROXIMITY_EVENT(event)                                                                             \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u)", \
                       (uint)event->pproximity.timestamp, (uint)event->pproximity.windowID, (uint)event->pproximity.which);
        SDL_EVENT_CASE(SDL_EVENT_PEN_PROXIMITY_IN)
        PRINT_PPROXIMITY_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_PEN_PROXIMITY_OUT)
        PRINT_PPROXIMITY_EVENT(event);
        break;
#undef PRINT_PPROXIMITY_EVENT

        SDL_EVENT_CASE(SDL_EVENT_PEN_AXIS)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u pen_state=%u x=%g y=%g axis=%s value=%g)",
                           (uint)event->paxis.timestamp, (uint)event->paxis.windowID, (uint)event->paxis.which, (uint)event->paxis.pen_state, event->paxis.x, event->paxis.y,
                           ((((int) event->paxis.axis) >= 0) && (event->paxis.axis < SDL_arraysize(pen_axisnames))) ? pen_axisnames[event->paxis.axis] : "[UNKNOWN]", event->paxis.value);
        break;

        SDL_EVENT_CASE(SDL_EVENT_PEN_MOTION)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u pen_state=%u x=%g y=%g)",
                           (uint)event->pmotion.timestamp, (uint)event->pmotion.windowID, (uint)event->pmotion.which, (uint)event->pmotion.pen_state, event->pmotion.x, event->pmotion.y);
        break;

#define PRINT_PBUTTON_EVENT(event)                                                                                                               \
    (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u windowid=%u which=%u pen_state=%u x=%g y=%g button=%u state=%s)", \
                       (uint)event->pbutton.timestamp, (uint)event->pbutton.windowID, (uint)event->pbutton.which, (uint)event->pbutton.pen_state, event->pbutton.x, event->pbutton.y, \
                       (uint)event->pbutton.button, event->pbutton.down ? "down" : "up");
        SDL_EVENT_CASE(SDL_EVENT_PEN_BUTTON_DOWN)
        PRINT_PBUTTON_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_PEN_BUTTON_UP)
        PRINT_PBUTTON_EVENT(event);
        break;
#undef PRINT_PBUTTON_EVENT

#define PRINT_DROP_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (data='%s' timestamp=%u windowid=%u x=%f y=%f)", event->drop.data, (uint)event->drop.timestamp, (uint)event->drop.windowID, event->drop.x, event->drop.y)
        SDL_EVENT_CASE(SDL_EVENT_DROP_FILE)
        PRINT_DROP_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_DROP_TEXT)
        PRINT_DROP_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_DROP_BEGIN)
        PRINT_DROP_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_DROP_COMPLETE)
        PRINT_DROP_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_DROP_POSITION)
        PRINT_DROP_EVENT(event);
        break;
#undef PRINT_DROP_EVENT

#define PRINT_AUDIODEV_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%u recording=%s)", (uint)event->adevice.timestamp, (uint)event->adevice.which, event->adevice.recording ? "true" : "false")
        SDL_EVENT_CASE(SDL_EVENT_AUDIO_DEVICE_ADDED)
        PRINT_AUDIODEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_AUDIO_DEVICE_REMOVED)
        PRINT_AUDIODEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_AUDIO_DEVICE_FORMAT_CHANGED)
        PRINT_AUDIODEV_EVENT(event);
        break;
#undef PRINT_AUDIODEV_EVENT

#define PRINT_CAMERADEV_EVENT(event) (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%u)", (uint)event->cdevice.timestamp, (uint)event->cdevice.which)
        SDL_EVENT_CASE(SDL_EVENT_CAMERA_DEVICE_ADDED)
        PRINT_CAMERADEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_CAMERA_DEVICE_REMOVED)
        PRINT_CAMERADEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_CAMERA_DEVICE_APPROVED)
        PRINT_CAMERADEV_EVENT(event);
        break;
        SDL_EVENT_CASE(SDL_EVENT_CAMERA_DEVICE_DENIED)
        PRINT_CAMERADEV_EVENT(event);
        break;
#undef PRINT_CAMERADEV_EVENT

        SDL_EVENT_CASE(SDL_EVENT_SENSOR_UPDATE)
        (void)SDL_snprintf(details, sizeof(details), " (timestamp=%u which=%d data[0]=%f data[1]=%f data[2]=%f data[3]=%f data[4]=%f data[5]=%f)",
                           (uint)event->sensor.timestamp, (int)event->sensor.which,
                           event->sensor.data[0], event->sensor.data[1], event->sensor.data[2],
                           event->sensor.data[3], event->sensor.data[4], event->sensor.data[5]);
        break;

#undef SDL_EVENT_CASE

    case SDL_EVENT_POLL_SENTINEL:
        // No logging necessary for this one
        break;

    default:
        if (!name[0]) {
            if (event->type >= SDL_EVENT_USER) {
                SDL_strlcpy(name, "USER", sizeof(name));
            } else {
                SDL_strlcpy(name, "UNKNOWN", sizeof(name));
            }
            (void)SDL_snprintf(details, sizeof(details), " 0x%x", (uint)event->type);
        }
        break;
    }

    if (name[0]) {
        SDL_Log("SDL EVENT: %s%s", name, details);
    }

#undef uint
}

void SDL_StopEventLoop(void)
{
    const char *report = SDL_GetHint("SDL_EVENT_QUEUE_STATISTICS");
    int i;
    SDL_EventEntry *entry;

    SDL_LockMutex(SDL_EventQ.lock);

    SDL_EventQ.active = false;

    if (report && SDL_atoi(report)) {
        SDL_Log("SDL EVENT QUEUE: Maximum events in-flight: %d",
                SDL_EventQ.max_events_seen);
    }

    // Clean out EventQ
    for (entry = SDL_EventQ.head; entry;) {
        SDL_EventEntry *next = entry->next;
        SDL_TransferTemporaryMemoryFromEvent(entry);
        SDL_free(entry);
        entry = next;
    }
    for (entry = SDL_EventQ.free; entry;) {
        SDL_EventEntry *next = entry->next;
        SDL_free(entry);
        entry = next;
    }

    SDL_SetAtomicInt(&SDL_EventQ.count, 0);
    SDL_EventQ.max_events_seen = 0;
    SDL_EventQ.head = NULL;
    SDL_EventQ.tail = NULL;
    SDL_EventQ.free = NULL;
    SDL_SetAtomicInt(&SDL_sentinel_pending, 0);

    // Clear disabled event state
    for (i = 0; i < SDL_arraysize(SDL_disabled_events); ++i) {
        SDL_free(SDL_disabled_events[i]);
        SDL_disabled_events[i] = NULL;
    }

    SDL_QuitEventWatchList(&SDL_event_watchers);
    //SDL_QuitWindowEventWatch();

    SDL_Mutex *lock = NULL;
    if (SDL_EventQ.lock) {
        lock = SDL_EventQ.lock;
        SDL_EventQ.lock = NULL;
    }

    SDL_UnlockMutex(lock);

    if (lock) {
        SDL_DestroyMutex(lock);
    }
}

// This function (and associated calls) may be called more than once
bool SDL_StartEventLoop(void)
{
    /* We'll leave the event queue alone, since we might have gotten
       some important events at launch (like SDL_EVENT_DROP_FILE)

       FIXME: Does this introduce any other bugs with events at startup?
     */

    // Create the lock and set ourselves active
#ifndef SDL_THREADS_DISABLED
    if (!SDL_EventQ.lock) {
        SDL_EventQ.lock = SDL_CreateMutex();
        if (SDL_EventQ.lock == NULL) {
            return false;
        }
    }
    SDL_LockMutex(SDL_EventQ.lock);

    if (!SDL_InitEventWatchList(&SDL_event_watchers)) {
        SDL_UnlockMutex(SDL_EventQ.lock);
        return false;
    }
#endif // !SDL_THREADS_DISABLED

    //SDL_InitWindowEventWatch();

    SDL_EventQ.active = true;

#ifndef SDL_THREADS_DISABLED
    SDL_UnlockMutex(SDL_EventQ.lock);
#endif
    return true;
}

// Add an event to the event queue -- called with the queue locked
static int SDL_AddEvent(SDL_Event *event)
{
    SDL_EventEntry *entry;
    const int initial_count = SDL_GetAtomicInt(&SDL_EventQ.count);
    int final_count;

    if (initial_count >= SDL_MAX_QUEUED_EVENTS) {
        SDL_SetError("Event queue is full (%d events)", initial_count);
        return 0;
    }

    if (SDL_EventQ.free == NULL) {
        entry = (SDL_EventEntry *)SDL_malloc(sizeof(*entry));
        if (entry == NULL) {
            return 0;
        }
    } else {
        entry = SDL_EventQ.free;
        SDL_EventQ.free = entry->next;
    }

    if (SDL_EventLoggingVerbosity > 0) {
        SDL_LogEvent(event);
    }

    SDL_copyp(&entry->event, event);
    if (event->type == SDL_EVENT_POLL_SENTINEL) {
        SDL_AddAtomicInt(&SDL_sentinel_pending, 1);
    }
    entry->memory = NULL;
    SDL_TransferTemporaryMemoryToEvent(entry);

    if (SDL_EventQ.tail) {
        SDL_EventQ.tail->next = entry;
        entry->prev = SDL_EventQ.tail;
        SDL_EventQ.tail = entry;
        entry->next = NULL;
    } else {
        SDL_assert(!SDL_EventQ.head);
        SDL_EventQ.head = entry;
        SDL_EventQ.tail = entry;
        entry->prev = NULL;
        entry->next = NULL;
    }

    final_count = SDL_AddAtomicInt(&SDL_EventQ.count, 1) + 1;
    if (final_count > SDL_EventQ.max_events_seen) {
        SDL_EventQ.max_events_seen = final_count;
    }

    ++SDL_last_event_id;

    return 1;
}

// Remove an event from the queue -- called with the queue locked
static void SDL_CutEvent(SDL_EventEntry *entry)
{
    SDL_TransferTemporaryMemoryFromEvent(entry);

    if (entry->prev) {
        entry->prev->next = entry->next;
    }
    if (entry->next) {
        entry->next->prev = entry->prev;
    }

    if (entry == SDL_EventQ.head) {
        SDL_assert(entry->prev == NULL);
        SDL_EventQ.head = entry->next;
    }
    if (entry == SDL_EventQ.tail) {
        SDL_assert(entry->next == NULL);
        SDL_EventQ.tail = entry->prev;
    }

    if (entry->event.type == SDL_EVENT_POLL_SENTINEL) {
        SDL_AddAtomicInt(&SDL_sentinel_pending, -1);
    }

    entry->next = SDL_EventQ.free;
    SDL_EventQ.free = entry;
    SDL_assert(SDL_GetAtomicInt(&SDL_EventQ.count) > 0);
    SDL_AddAtomicInt(&SDL_EventQ.count, -1);
}

static void SDL_SendWakeupEvent(void)
{
#ifdef SDL_PLATFORM_ANDROID
    Android_SendLifecycleEvent(SDL_ANDROID_LIFECYCLE_WAKE);
#endif
}

// Lock the event queue, take a peep at it, and unlock it
static int SDL_PeepEventsInternal(SDL_Event *events, int numevents, SDL_EventAction action,
                                  Uint32 minType, Uint32 maxType, bool include_sentinel)
{
    int i, used, sentinels_expected = 0;

    // Lock the event queue
    used = 0;

    SDL_LockMutex(SDL_EventQ.lock);
    {
        // Don't look after we've quit
        if (!SDL_EventQ.active) {
            // We get a few spurious events at shutdown, so don't warn then
            if (action == SDL_GETEVENT) {
                SDL_SetError("The event system has been shut down");
            }
            SDL_UnlockMutex(SDL_EventQ.lock);
            return -1;
        }
        if (action == SDL_ADDEVENT) {
            if (!events) {
                SDL_UnlockMutex(SDL_EventQ.lock);
                SDL_InvalidParamError("events");
                return -1;
            }
            for (i = 0; i < numevents; ++i) {
                used += SDL_AddEvent(&events[i]);
            }
        } else {
            SDL_EventEntry *entry, *next;
            Uint32 type;

            for (entry = SDL_EventQ.head; entry && (events == NULL || used < numevents); entry = next) {
                next = entry->next;
                type = entry->event.type;
                if (minType <= type && type <= maxType) {
                    if (events) {
                        SDL_copyp(&events[used], &entry->event);

                        if (action == SDL_GETEVENT) {
                            SDL_CutEvent(entry);
                        }
                    }
                    if (type == SDL_EVENT_POLL_SENTINEL) {
                        // Special handling for the sentinel event
                        if (!include_sentinel) {
                            // Skip it, we don't want to include it
                            continue;
                        }
                        if (events == NULL || action != SDL_GETEVENT) {
                            ++sentinels_expected;
                        }
                        if (SDL_GetAtomicInt(&SDL_sentinel_pending) > sentinels_expected) {
                            // Skip it, there's another one pending
                            continue;
                        }
                    }
                    ++used;
                }
            }
        }
    }
    SDL_UnlockMutex(SDL_EventQ.lock);

    if (used > 0 && action == SDL_ADDEVENT) {
        SDL_SendWakeupEvent();
    }

    return used;
}
int SDL_PeepEvents(SDL_Event *events, int numevents, SDL_EventAction action,
                   Uint32 minType, Uint32 maxType)
{
    return SDL_PeepEventsInternal(events, numevents, action, minType, maxType, false);
}

bool SDL_HasEvent(Uint32 type)
{
    return SDL_HasEvents(type, type);
}

bool SDL_HasEvents(Uint32 minType, Uint32 maxType)
{
    bool found = false;

    SDL_LockMutex(SDL_EventQ.lock);
    {
        if (SDL_EventQ.active) {
            for (SDL_EventEntry *entry = SDL_EventQ.head; entry; entry = entry->next) {
                const Uint32 type = entry->event.type;
                if (minType <= type && type <= maxType) {
                    found = true;
                    break;
                }
            }
        }
    }
    SDL_UnlockMutex(SDL_EventQ.lock);

    return found;
}

void SDL_FlushEvent(Uint32 type)
{
    SDL_FlushEvents(type, type);
}

void SDL_FlushEvents(Uint32 minType, Uint32 maxType)
{
    SDL_EventEntry *entry, *next;
    Uint32 type;

    // Make sure the events are current
#if 0
    /* Actually, we can't do this since we might be flushing while processing
       a resize event, and calling this might trigger further resize events.
    */
    SDL_PumpEvents();
#endif

    // Lock the event queue
    SDL_LockMutex(SDL_EventQ.lock);
    {
        // Don't look after we've quit
        if (!SDL_EventQ.active) {
            SDL_UnlockMutex(SDL_EventQ.lock);
            return;
        }
        for (entry = SDL_EventQ.head; entry; entry = next) {
            next = entry->next;
            type = entry->event.type;
            if (minType <= type && type <= maxType) {
                SDL_CutEvent(entry);
            }
        }
    }
    SDL_UnlockMutex(SDL_EventQ.lock);
}

typedef enum
{
    SDL_MAIN_CALLBACK_WAITING,
    SDL_MAIN_CALLBACK_COMPLETE,
    SDL_MAIN_CALLBACK_CANCELED,
} SDL_MainThreadCallbackState;

typedef struct SDL_MainThreadCallbackEntry
{
    SDL_MainThreadCallback callback;
    void *userdata;
    SDL_AtomicInt state;
    SDL_Semaphore *semaphore;
    struct SDL_MainThreadCallbackEntry *next;
} SDL_MainThreadCallbackEntry;

static SDL_Mutex *SDL_main_callbacks_lock;
static SDL_MainThreadCallbackEntry *SDL_main_callbacks_head;
static SDL_MainThreadCallbackEntry *SDL_main_callbacks_tail;

static SDL_MainThreadCallbackEntry *SDL_CreateMainThreadCallback(SDL_MainThreadCallback callback, void *userdata, bool wait_complete)
{
    SDL_MainThreadCallbackEntry *entry = (SDL_MainThreadCallbackEntry *)SDL_malloc(sizeof(*entry));
    if (!entry) {
        return NULL;
    }

    entry->callback = callback;
    entry->userdata = userdata;
    SDL_SetAtomicInt(&entry->state, SDL_MAIN_CALLBACK_WAITING);
    if (wait_complete) {
        entry->semaphore = SDL_CreateSemaphore(0);
        if (!entry->semaphore) {
            SDL_free(entry);
            return NULL;
        }
    } else {
        entry->semaphore = NULL;
    }
    entry->next = NULL;

    return entry;
}

static void SDL_DestroyMainThreadCallback(SDL_MainThreadCallbackEntry *entry)
{
    if (entry->semaphore) {
        SDL_DestroySemaphore(entry->semaphore);
    }
    SDL_free(entry);
}

static void SDL_InitMainThreadCallbacks(void)
{
    SDL_main_callbacks_lock = SDL_CreateMutex();
    SDL_assert(SDL_main_callbacks_head == NULL &&
               SDL_main_callbacks_tail == NULL);
}

static void SDL_QuitMainThreadCallbacks(void)
{
    SDL_MainThreadCallbackEntry *entry;

    SDL_LockMutex(SDL_main_callbacks_lock);
    {
        entry = SDL_main_callbacks_head;
        SDL_main_callbacks_head = NULL;
        SDL_main_callbacks_tail = NULL;
    }
    SDL_UnlockMutex(SDL_main_callbacks_lock);

    while (entry) {
        SDL_MainThreadCallbackEntry *next = entry->next;

        if (entry->semaphore) {
            // Let the waiting thread know this is canceled
            SDL_SetAtomicInt(&entry->state, SDL_MAIN_CALLBACK_CANCELED);
            SDL_SignalSemaphore(entry->semaphore);
        } else {
            // Nobody's waiting for this, clean it up
            SDL_DestroyMainThreadCallback(entry);
        }
        entry = next;
    }

    SDL_DestroyMutex(SDL_main_callbacks_lock);
    SDL_main_callbacks_lock = NULL;
}

static void SDL_RunMainThreadCallbacks(void)
{
    SDL_MainThreadCallbackEntry *entry;

    SDL_LockMutex(SDL_main_callbacks_lock);
    {
        entry = SDL_main_callbacks_head;
        SDL_main_callbacks_head = NULL;
        SDL_main_callbacks_tail = NULL;
    }
    SDL_UnlockMutex(SDL_main_callbacks_lock);

    while (entry) {
        SDL_MainThreadCallbackEntry *next = entry->next;

        entry->callback(entry->userdata);

        if (entry->semaphore) {
            // Let the waiting thread know this is done
            SDL_SetAtomicInt(&entry->state, SDL_MAIN_CALLBACK_COMPLETE);
            SDL_SignalSemaphore(entry->semaphore);
        } else {
            // Nobody's waiting for this, clean it up
            SDL_DestroyMainThreadCallback(entry);
        }
        entry = next;
    }
}

bool SDL_RunOnMainThread(SDL_MainThreadCallback callback, void *userdata, bool wait_complete)
{
    if (SDL_IsMainThread() || !SDL_WasInit(SDL_INIT_EVENTS)) {
        // No need to queue the callback
        callback(userdata);
        return true;
    }

    SDL_MainThreadCallbackEntry *entry = SDL_CreateMainThreadCallback(callback, userdata, wait_complete);
    if (!entry) {
        return false;
    }

    SDL_LockMutex(SDL_main_callbacks_lock);
    {
        if (SDL_main_callbacks_tail) {
            SDL_main_callbacks_tail->next = entry;
            SDL_main_callbacks_tail = entry;
        } else {
            SDL_main_callbacks_head = entry;
            SDL_main_callbacks_tail = entry;
        }
    }
    SDL_UnlockMutex(SDL_main_callbacks_lock);

    // If the main thread is waiting for events, wake it up
    SDL_SendWakeupEvent();

    if (!wait_complete) {
        // Queued for execution, wait not requested
        return true;
    }

    SDL_WaitSemaphore(entry->semaphore);

    switch (SDL_GetAtomicInt(&entry->state)) {
    case SDL_MAIN_CALLBACK_COMPLETE:
        // Execution complete!
        SDL_DestroyMainThreadCallback(entry);
        return true;

    case SDL_MAIN_CALLBACK_CANCELED:
        // The callback was canceled on the main thread
        SDL_DestroyMainThreadCallback(entry);
        return SDL_SetError("Callback canceled");

    default:
        // Probably hit a deadlock in the callback
        // We can't destroy the entry as the semaphore will be signaled
        // if it ever comes back, just leak it here.
        return SDL_SetError("Callback timed out");
    }
}

void SDL_PumpEventMaintenance(void)
{
#ifndef SDL_AUDIO_DISABLED
    SDL_UpdateAudio();
#endif

#ifndef SDL_CAMERA_DISABLED
    SDL_UpdateCamera();
#endif

#ifndef SDL_SENSOR_DISABLED
    // Check for sensor state change
    if (SDL_update_sensors) {
        SDL_UpdateSensors();
    }
#endif

#ifndef SDL_JOYSTICK_DISABLED
    // Check for joystick state change
    if (SDL_update_joysticks) {
        SDL_UpdateJoysticks();
    }
#endif

    //SDL_UpdateTrays();

    //SDL_SendPendingSignalEvents(); // in case we had a signal handler fire, etc.
}

// Run the system dependent event loops
static void SDL_PumpEventsInternal(bool push_sentinel)
{
    // Free any temporary memory from old events
    SDL_FreeTemporaryMemory();

    // Release any keys held down from last frame
    //SDL_ReleaseAutoReleaseKeys();

    // Run any pending main thread callbacks
    SDL_RunMainThreadCallbacks();

#ifdef SDL_PLATFORM_ANDROID
    // Android event processing is independent of the video subsystem
    Android_PumpEvents(0);
#endif

    SDL_PumpEventMaintenance();

    if (push_sentinel && SDL_EventEnabled(SDL_EVENT_POLL_SENTINEL)) {
        SDL_Event sentinel;

        // Make sure we don't already have a sentinel in the queue, and add one to the end
        if (SDL_GetAtomicInt(&SDL_sentinel_pending) > 0) {
            SDL_PeepEventsInternal(&sentinel, 1, SDL_GETEVENT, SDL_EVENT_POLL_SENTINEL, SDL_EVENT_POLL_SENTINEL, true);
        }

        sentinel.type = SDL_EVENT_POLL_SENTINEL;
        sentinel.common.timestamp = 0;
        SDL_PushEvent(&sentinel);
    }
}

void SDL_PumpEvents(void)
{
    SDL_PumpEventsInternal(false);
}

// Public functions

bool SDL_PollEvent(SDL_Event *event)
{
    return SDL_WaitEventTimeoutNS(event, 0);
}

bool SDL_WaitEvent(SDL_Event *event)
{
    return SDL_WaitEventTimeoutNS(event, -1);
}

bool SDL_WaitEventTimeout(SDL_Event *event, Sint32 timeoutMS)
{
    Sint64 timeoutNS;

    if (timeoutMS > 0) {
        timeoutNS = SDL_MS_TO_NS(timeoutMS);
    } else {
        timeoutNS = timeoutMS;
    }
    return SDL_WaitEventTimeoutNS(event, timeoutNS);
}

bool SDL_WaitEventTimeoutNS(SDL_Event *event, Sint64 timeoutNS)
{
    Uint64 start, expiration;
    bool include_sentinel = (timeoutNS == 0);
    int result;

    if (timeoutNS > 0) {
        start = SDL_GetTicksNS();
        expiration = start + timeoutNS;
    } else {
        start = 0;
        expiration = 0;
    }

    // If there isn't a poll sentinel event pending, pump events and add one
    if (SDL_GetAtomicInt(&SDL_sentinel_pending) == 0) {
        SDL_PumpEventsInternal(true);
    }

    // First check for existing events
    result = SDL_PeepEventsInternal(event, 1, SDL_GETEVENT, SDL_EVENT_FIRST, SDL_EVENT_LAST, include_sentinel);
    if (result < 0) {
        return false;
    }
    if (include_sentinel) {
        if (event) {
            if (event->type == SDL_EVENT_POLL_SENTINEL) {
                // Reached the end of a poll cycle, and not willing to wait
                return false;
            }
        } else {
            // Need to peek the next event to check for sentinel
            SDL_Event dummy;

            if (SDL_PeepEventsInternal(&dummy, 1, SDL_PEEKEVENT, SDL_EVENT_FIRST, SDL_EVENT_LAST, true) &&
                dummy.type == SDL_EVENT_POLL_SENTINEL) {
                SDL_PeepEventsInternal(&dummy, 1, SDL_GETEVENT, SDL_EVENT_POLL_SENTINEL, SDL_EVENT_POLL_SENTINEL, true);
                // Reached the end of a poll cycle, and not willing to wait
                return false;
            }
        }
    }
    if (result == 0) {
        if (timeoutNS == 0) {
            // No events available, and not willing to wait
            return false;
        }
    } else {
        // Has existing events
        return true;
    }
    // We should have completely handled timeoutNS == 0 above
    SDL_assert(timeoutNS != 0);

#ifdef SDL_PLATFORM_ANDROID
    for (;;) {
        SDL_PumpEventsInternal(true);

        if (SDL_PeepEvents(event, 1, SDL_GETEVENT, SDL_EVENT_FIRST, SDL_EVENT_LAST) > 0) {
            return true;
        }

        Uint64 delay = -1;
        if (timeoutNS > 0) {
            Uint64 now = SDL_GetTicksNS();
            if (now >= expiration) {
                // Timeout expired and no events
                return false;
            }
            delay = (expiration - now);
        }
        Android_PumpEvents(delay);
    }
#else
    for (;;) {
        SDL_PumpEventsInternal(true);

        if (SDL_PeepEvents(event, 1, SDL_GETEVENT, SDL_EVENT_FIRST, SDL_EVENT_LAST) > 0) {
            return true;
        }

        Uint64 delay = EVENT_POLL_INTERVAL_NS;
        if (timeoutNS > 0) {
            Uint64 now = SDL_GetTicksNS();
            if (now >= expiration) {
                // Timeout expired and no events
                return false;
            }
            delay = SDL_min((expiration - now), delay);
        }
        SDL_DelayNS(delay);
    }
#endif // SDL_PLATFORM_ANDROID
}

static bool SDL_CallEventWatchers(SDL_Event *event)
{
    if (event->common.type == SDL_EVENT_POLL_SENTINEL) {
        return true;
    }

    return SDL_DispatchEventWatchList(&SDL_event_watchers, event);
}

bool SDL_PushEvent(SDL_Event *event)
{
    if (!event->common.timestamp) {
        event->common.timestamp = SDL_GetTicksNS();
    }

    if (!SDL_CallEventWatchers(event)) {
        SDL_ClearError();
        return false;
    }

    if (SDL_PeepEvents(event, 1, SDL_ADDEVENT, 0, 0) <= 0) {
        return false;
    }

    return true;
}

void SDL_SetEventFilter(SDL_EventFilter filter, void *userdata)
{
    SDL_EventEntry *event, *next;
    SDL_LockMutex(SDL_event_watchers.lock);
    {
        // Set filter and discard pending events
        SDL_event_watchers.filter.callback = filter;
        SDL_event_watchers.filter.userdata = userdata;
        if (filter) {
            // Cut all events not accepted by the filter
            SDL_LockMutex(SDL_EventQ.lock);
            {
                for (event = SDL_EventQ.head; event; event = next) {
                    next = event->next;
                    if (!filter(userdata, &event->event)) {
                        SDL_CutEvent(event);
                    }
                }
            }
            SDL_UnlockMutex(SDL_EventQ.lock);
        }
    }
    SDL_UnlockMutex(SDL_event_watchers.lock);
}

bool SDL_GetEventFilter(SDL_EventFilter *filter, void **userdata)
{
    SDL_EventWatcher event_ok;

    SDL_LockMutex(SDL_event_watchers.lock);
    {
        event_ok = SDL_event_watchers.filter;
    }
    SDL_UnlockMutex(SDL_event_watchers.lock);

    if (filter) {
        *filter = event_ok.callback;
    }
    if (userdata) {
        *userdata = event_ok.userdata;
    }
    return event_ok.callback ? true : false;
}

bool SDL_AddEventWatch(SDL_EventFilter filter, void *userdata)
{
    return SDL_AddEventWatchList(&SDL_event_watchers, filter, userdata);
}

void SDL_RemoveEventWatch(SDL_EventFilter filter, void *userdata)
{
    SDL_RemoveEventWatchList(&SDL_event_watchers, filter, userdata);
}

void SDL_FilterEvents(SDL_EventFilter filter, void *userdata)
{
    SDL_LockMutex(SDL_EventQ.lock);
    {
        SDL_EventEntry *entry, *next;
        for (entry = SDL_EventQ.head; entry; entry = next) {
            next = entry->next;
            if (!filter(userdata, &entry->event)) {
                SDL_CutEvent(entry);
            }
        }
    }
    SDL_UnlockMutex(SDL_EventQ.lock);
}

void SDL_SetEventEnabled(Uint32 type, bool enabled)
{
    bool current_state;
    Uint8 hi = ((type >> 8) & 0xff);
    Uint8 lo = (type & 0xff);

    if (SDL_disabled_events[hi] &&
        (SDL_disabled_events[hi]->bits[lo / 32] & (1U << (lo & 31)))) {
        current_state = false;
    } else {
        current_state = true;
    }

    if ((enabled != false) != current_state) {
        if (enabled) {
            SDL_assert(SDL_disabled_events[hi] != NULL);
            SDL_disabled_events[hi]->bits[lo / 32] &= ~(1U << (lo & 31));

            // Gamepad events depend on joystick events
            switch (type) {
            case SDL_EVENT_GAMEPAD_ADDED:
                SDL_SetEventEnabled(SDL_EVENT_JOYSTICK_ADDED, true);
                break;
            case SDL_EVENT_GAMEPAD_REMOVED:
                SDL_SetEventEnabled(SDL_EVENT_JOYSTICK_REMOVED, true);
                break;
            case SDL_EVENT_GAMEPAD_AXIS_MOTION:
            case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
            case SDL_EVENT_GAMEPAD_BUTTON_UP:
                SDL_SetEventEnabled(SDL_EVENT_JOYSTICK_AXIS_MOTION, true);
                SDL_SetEventEnabled(SDL_EVENT_JOYSTICK_HAT_MOTION, true);
                SDL_SetEventEnabled(SDL_EVENT_JOYSTICK_BUTTON_DOWN, true);
                SDL_SetEventEnabled(SDL_EVENT_JOYSTICK_BUTTON_UP, true);
                break;
            case SDL_EVENT_GAMEPAD_UPDATE_COMPLETE:
                SDL_SetEventEnabled(SDL_EVENT_JOYSTICK_UPDATE_COMPLETE, true);
                break;
            default:
                break;
            }
        } else {
            // Disable this event type and discard pending events
            if (!SDL_disabled_events[hi]) {
                SDL_disabled_events[hi] = (SDL_DisabledEventBlock *)SDL_calloc(1, sizeof(SDL_DisabledEventBlock));
            }
            // Out of memory, nothing we can do...
            if (SDL_disabled_events[hi]) {
                SDL_disabled_events[hi]->bits[lo / 32] |= (1U << (lo & 31));
                SDL_FlushEvent(type);
            }
        }

        /* turn off drag'n'drop support if we've disabled the events.
           This might change some UI details at the OS level. */
        if (type == SDL_EVENT_DROP_FILE || type == SDL_EVENT_DROP_TEXT) {
            //SDL_ToggleDragAndDropSupport();
        }
    }
}

bool SDL_EventEnabled(Uint32 type)
{
    Uint8 hi = ((type >> 8) & 0xff);
    Uint8 lo = (type & 0xff);

    if (SDL_disabled_events[hi] &&
        (SDL_disabled_events[hi]->bits[lo / 32] & (1U << (lo & 31)))) {
        return false;
    } else {
        return true;
    }
}

Uint32 SDL_RegisterEvents(int numevents)
{
    Uint32 event_base = 0;

    if (numevents > 0) {
        int value = SDL_AddAtomicInt(&SDL_userevents, numevents);
        if (value >= 0 && value <= (SDL_EVENT_LAST - SDL_EVENT_USER)) {
            event_base = (Uint32)(SDL_EVENT_USER + value);
        }
    }
    return event_base;
}

void SDL_SendAppEvent(SDL_EventType eventType)
{
    if (SDL_EventEnabled(eventType)) {
        SDL_Event event;
        event.type = eventType;
        event.common.timestamp = 0;

        switch (eventType) {
        case SDL_EVENT_TERMINATING:
        case SDL_EVENT_LOW_MEMORY:
        case SDL_EVENT_WILL_ENTER_BACKGROUND:
        case SDL_EVENT_DID_ENTER_BACKGROUND:
        case SDL_EVENT_WILL_ENTER_FOREGROUND:
        case SDL_EVENT_DID_ENTER_FOREGROUND:
            // We won't actually queue this event, it needs to be handled in this call stack by an event watcher
            if (SDL_EventLoggingVerbosity > 0) {
                SDL_LogEvent(&event);
            }
            SDL_CallEventWatchers(&event);
            break;
        default:
            SDL_PushEvent(&event);
            break;
        }
    }
}

void SDL_SendKeymapChangedEvent(void)
{
    SDL_SendAppEvent(SDL_EVENT_KEYMAP_CHANGED);
}

void SDL_SendLocaleChangedEvent(void)
{
    SDL_SendAppEvent(SDL_EVENT_LOCALE_CHANGED);
}

void SDL_SendSystemThemeChangedEvent(void)
{
    SDL_SendAppEvent(SDL_EVENT_SYSTEM_THEME_CHANGED);
}

bool SDL_InitEvents(void)
{
#ifdef SDL_PLATFORM_ANDROID
    Android_InitEvents();
#endif
#ifndef SDL_JOYSTICK_DISABLED
    SDL_AddHintCallback(SDL_HINT_AUTO_UPDATE_JOYSTICKS, SDL_AutoUpdateJoysticksChanged, NULL);
#endif
#ifndef SDL_SENSOR_DISABLED
    SDL_AddHintCallback(SDL_HINT_AUTO_UPDATE_SENSORS, SDL_AutoUpdateSensorsChanged, NULL);
#endif
    SDL_AddHintCallback(SDL_HINT_EVENT_LOGGING, SDL_EventLoggingChanged, NULL);
    SDL_AddHintCallback(SDL_HINT_POLL_SENTINEL, SDL_PollSentinelChanged, NULL);
    SDL_InitMainThreadCallbacks();
    if (!SDL_StartEventLoop()) {
        SDL_RemoveHintCallback(SDL_HINT_EVENT_LOGGING, SDL_EventLoggingChanged, NULL);
        return false;
    }

    //SDL_InitQuit();

    return true;
}

void SDL_QuitEvents(void)
{
    //SDL_QuitQuit();
    SDL_StopEventLoop();
    SDL_QuitMainThreadCallbacks();
    SDL_RemoveHintCallback(SDL_HINT_POLL_SENTINEL, SDL_PollSentinelChanged, NULL);
    SDL_RemoveHintCallback(SDL_HINT_EVENT_LOGGING, SDL_EventLoggingChanged, NULL);
#ifndef SDL_JOYSTICK_DISABLED
    SDL_RemoveHintCallback(SDL_HINT_AUTO_UPDATE_JOYSTICKS, SDL_AutoUpdateJoysticksChanged, NULL);
#endif
#ifndef SDL_SENSOR_DISABLED
    SDL_RemoveHintCallback(SDL_HINT_AUTO_UPDATE_SENSORS, SDL_AutoUpdateSensorsChanged, NULL);
#endif
#ifdef SDL_PLATFORM_ANDROID
    Android_QuitEvents();
#endif
}
