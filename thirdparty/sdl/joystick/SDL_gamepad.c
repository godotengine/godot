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

// This is the gamepad API for Simple DirectMedia Layer

#include "SDL_sysjoystick.h"
#include "SDL_joystick_c.h"
#include "SDL_steam_virtual_gamepad.h"
#include "SDL_gamepad_c.h"
#include "SDL_gamepad_db.h"
#include "controller_type.h"
#include "usb_ids.h"
#include "hidapi/SDL_hidapi_nintendo.h"
#include "../events/SDL_events_c.h"

#ifdef SDL_PLATFORM_WIN32
#include "../core/windows/SDL_windows.h"
#endif


// Many gamepads turn the center button into an instantaneous button press
#define SDL_MINIMUM_GUIDE_BUTTON_DELAY_MS 250

#define SDL_GAMEPAD_CRC_FIELD           "crc:"
#define SDL_GAMEPAD_CRC_FIELD_SIZE      4 // hard-coded for speed
#define SDL_GAMEPAD_TYPE_FIELD          "type:"
#define SDL_GAMEPAD_TYPE_FIELD_SIZE     SDL_strlen(SDL_GAMEPAD_TYPE_FIELD)
#define SDL_GAMEPAD_FACE_FIELD          "face:"
#define SDL_GAMEPAD_FACE_FIELD_SIZE     5 // hard-coded for speed
#define SDL_GAMEPAD_PLATFORM_FIELD      "platform:"
#define SDL_GAMEPAD_PLATFORM_FIELD_SIZE SDL_strlen(SDL_GAMEPAD_PLATFORM_FIELD)
#define SDL_GAMEPAD_HINT_FIELD          "hint:"
#define SDL_GAMEPAD_HINT_FIELD_SIZE     SDL_strlen(SDL_GAMEPAD_HINT_FIELD)
#define SDL_GAMEPAD_SDKGE_FIELD         "sdk>=:"
#define SDL_GAMEPAD_SDKGE_FIELD_SIZE    SDL_strlen(SDL_GAMEPAD_SDKGE_FIELD)
#define SDL_GAMEPAD_SDKLE_FIELD         "sdk<=:"
#define SDL_GAMEPAD_SDKLE_FIELD_SIZE    SDL_strlen(SDL_GAMEPAD_SDKLE_FIELD)

static bool SDL_gamepads_initialized;
static SDL_Gamepad *SDL_gamepads SDL_GUARDED_BY(SDL_joystick_lock) = NULL;

// The face button style of a gamepad
typedef enum
{
    SDL_GAMEPAD_FACE_STYLE_UNKNOWN,
    SDL_GAMEPAD_FACE_STYLE_ABXY,
    SDL_GAMEPAD_FACE_STYLE_BAYX,
    SDL_GAMEPAD_FACE_STYLE_SONY,
} SDL_GamepadFaceStyle;

// our hard coded list of mapping support
typedef enum
{
    SDL_GAMEPAD_MAPPING_PRIORITY_DEFAULT,
    SDL_GAMEPAD_MAPPING_PRIORITY_API,
    SDL_GAMEPAD_MAPPING_PRIORITY_USER,
} SDL_GamepadMappingPriority;

#define _guarded SDL_GUARDED_BY(SDL_joystick_lock)

typedef struct GamepadMapping_t
{
    SDL_GUID guid _guarded;
    char *name _guarded;
    char *mapping _guarded;
    SDL_GamepadMappingPriority priority _guarded;
    struct GamepadMapping_t *next _guarded;
} GamepadMapping_t;

typedef struct
{
    int refcount _guarded;
    SDL_JoystickID *joysticks _guarded;
    GamepadMapping_t **joystick_mappings _guarded;

    int num_changed_mappings _guarded;
    GamepadMapping_t **changed_mappings _guarded;

} MappingChangeTracker;

#undef _guarded

static SDL_GUID s_zeroGUID;
static GamepadMapping_t *s_pSupportedGamepads SDL_GUARDED_BY(SDL_joystick_lock) = NULL;
static GamepadMapping_t *s_pDefaultMapping SDL_GUARDED_BY(SDL_joystick_lock) = NULL;
static GamepadMapping_t *s_pXInputMapping SDL_GUARDED_BY(SDL_joystick_lock) = NULL;
static MappingChangeTracker *s_mappingChangeTracker SDL_GUARDED_BY(SDL_joystick_lock) = NULL;
static SDL_HashTable *s_gamepadInstanceIDs SDL_GUARDED_BY(SDL_joystick_lock) = NULL;

#define _guarded SDL_GUARDED_BY(SDL_joystick_lock)

// The SDL gamepad structure
struct SDL_Gamepad
{
    SDL_Joystick *joystick _guarded; // underlying joystick device
    int ref_count _guarded;

    const char *name _guarded;
    SDL_GamepadType type _guarded;
    SDL_GamepadFaceStyle face_style _guarded;
    GamepadMapping_t *mapping _guarded;
    int num_bindings _guarded;
    SDL_GamepadBinding *bindings _guarded;
    SDL_GamepadBinding **last_match_axis _guarded;
    Uint8 *last_hat_mask _guarded;
    Uint64 guide_button_down _guarded;

    struct SDL_Gamepad *next _guarded; // pointer to next gamepad we have allocated
};

#undef _guarded

#define CHECK_GAMEPAD_MAGIC(gamepad, result)                    \
    if (!SDL_ObjectValid(gamepad, SDL_OBJECT_TYPE_GAMEPAD) ||   \
        !SDL_IsJoystickValid(gamepad->joystick)) {              \
        SDL_InvalidParamError("gamepad");                       \
        SDL_UnlockJoysticks();                                  \
        return result;                                          \
    }

static SDL_vidpid_list SDL_allowed_gamepads = {
    SDL_HINT_GAMECONTROLLER_IGNORE_DEVICES_EXCEPT, 0, 0, NULL,
    NULL, 0, 0, NULL,
    0, NULL,
    false
};
static SDL_vidpid_list SDL_ignored_gamepads = {
    SDL_HINT_GAMECONTROLLER_IGNORE_DEVICES, 0, 0, NULL,
    NULL, 0, 0, NULL,
    0, NULL,
    false
};

/*
    List of words in gamepad names that indicate that the gamepad should not be detected.
    See also `initial_blacklist_devices` in SDL_joystick.c
*/

enum SDL_GamepadBlacklistWordsPosition {
    GAMEPAD_BLACKLIST_BEGIN,
    GAMEPAD_BLACKLIST_END,
    GAMEPAD_BLACKLIST_ANYWHERE,
};

struct SDL_GamepadBlacklistWords {
    const char* str;
    enum SDL_GamepadBlacklistWordsPosition pos;
};

static const struct SDL_GamepadBlacklistWords SDL_gamepad_blacklist_words[] = {
#ifdef SDL_PLATFORM_LINUX
    {" Motion Sensors", GAMEPAD_BLACKLIST_END}, // Don't treat the PS3 and PS4 motion controls as a separate gamepad
    {" IMU",            GAMEPAD_BLACKLIST_END}, // Don't treat the Nintendo IMU as a separate gamepad
    {" Touchpad",       GAMEPAD_BLACKLIST_END}, // "Sony Interactive Entertainment DualSense Wireless Controller Touchpad"

    // Don't treat the Wii extension controls as a separate gamepad
    {" Accelerometer",  GAMEPAD_BLACKLIST_END},
    {" IR",             GAMEPAD_BLACKLIST_END},
    {" Motion Plus",    GAMEPAD_BLACKLIST_END},
    {" Nunchuk",        GAMEPAD_BLACKLIST_END},
#endif

    // The Google Pixel fingerprint sensor, as well as other fingerprint sensors, reports itself as a joystick
    {"uinput-",         GAMEPAD_BLACKLIST_BEGIN},

    {"Synaptics ",      GAMEPAD_BLACKLIST_ANYWHERE}, // "Synaptics TM2768-001", "SynPS/2 Synaptics TouchPad"
    {"Trackpad",        GAMEPAD_BLACKLIST_ANYWHERE},
    {"Clickpad",        GAMEPAD_BLACKLIST_ANYWHERE},
    // "PG-90215 Keyboard", "Usb Keyboard Usb Keyboard Consumer Control", "Framework Laptop 16 Keyboard Module - ISO System Control"
    {" Keyboard",       GAMEPAD_BLACKLIST_ANYWHERE},
    {" Laptop ",        GAMEPAD_BLACKLIST_ANYWHERE}, // "Framework Laptop 16 Numpad Module System Control"
    {"Mouse ",          GAMEPAD_BLACKLIST_BEGIN}, // "Mouse passthrough"
    {" Pen",            GAMEPAD_BLACKLIST_END}, // "Wacom One by Wacom S Pen"
    {" Finger",         GAMEPAD_BLACKLIST_END}, // "Wacom HID 495F Finger"
    {" LED ",           GAMEPAD_BLACKLIST_ANYWHERE}, // "ASRock LED Controller"
    {" Thelio ",        GAMEPAD_BLACKLIST_ANYWHERE}, // "System76 Thelio Io 2"
};

static GamepadMapping_t *SDL_PrivateAddMappingForGUID(SDL_GUID jGUID, const char *mappingString, bool *existing, SDL_GamepadMappingPriority priority);
static void SDL_PrivateLoadButtonMapping(SDL_Gamepad *gamepad, GamepadMapping_t *pGamepadMapping);
static GamepadMapping_t *SDL_PrivateGetGamepadMapping(SDL_JoystickID instance_id, bool create_mapping);
static void SDL_SendGamepadAxis(Uint64 timestamp, SDL_Gamepad *gamepad, SDL_GamepadAxis axis, Sint16 value);
static void SDL_SendGamepadButton(Uint64 timestamp, SDL_Gamepad *gamepad, SDL_GamepadButton button, bool down);

static bool HasSameOutput(SDL_GamepadBinding *a, SDL_GamepadBinding *b)
{
    if (a->output_type != b->output_type) {
        return false;
    }

    if (a->output_type == SDL_GAMEPAD_BINDTYPE_AXIS) {
        return a->output.axis.axis == b->output.axis.axis;
    } else {
        return a->output.button == b->output.button;
    }
}

static void ResetOutput(Uint64 timestamp, SDL_Gamepad *gamepad, SDL_GamepadBinding *bind)
{
    if (bind->output_type == SDL_GAMEPAD_BINDTYPE_AXIS) {
        SDL_SendGamepadAxis(timestamp, gamepad, bind->output.axis.axis, 0);
    } else {
        SDL_SendGamepadButton(timestamp, gamepad, bind->output.button, false);
    }
}

static void HandleJoystickAxis(Uint64 timestamp, SDL_Gamepad *gamepad, int axis, int value)
{
    int i;
    SDL_GamepadBinding *last_match;
    SDL_GamepadBinding *match = NULL;

    SDL_AssertJoysticksLocked();

    last_match = gamepad->last_match_axis[axis];
    for (i = 0; i < gamepad->num_bindings; ++i) {
        SDL_GamepadBinding *binding = &gamepad->bindings[i];
        if (binding->input_type == SDL_GAMEPAD_BINDTYPE_AXIS &&
            axis == binding->input.axis.axis) {
            if (binding->input.axis.axis_min < binding->input.axis.axis_max) {
                if (value >= binding->input.axis.axis_min &&
                    value <= binding->input.axis.axis_max) {
                    match = binding;
                    break;
                }
            } else {
                if (value >= binding->input.axis.axis_max &&
                    value <= binding->input.axis.axis_min) {
                    match = binding;
                    break;
                }
            }
        }
    }

    if (last_match && (!match || !HasSameOutput(last_match, match))) {
        // Clear the last input that this axis generated
        ResetOutput(timestamp, gamepad, last_match);
    }

    if (match) {
        if (match->output_type == SDL_GAMEPAD_BINDTYPE_AXIS) {
            if (match->input.axis.axis_min != match->output.axis.axis_min || match->input.axis.axis_max != match->output.axis.axis_max) {
                float normalized_value = (float)(value - match->input.axis.axis_min) / (match->input.axis.axis_max - match->input.axis.axis_min);
                value = match->output.axis.axis_min + (int)(normalized_value * (match->output.axis.axis_max - match->output.axis.axis_min));
            }
            SDL_SendGamepadAxis(timestamp, gamepad, match->output.axis.axis, (Sint16)value);
        } else {
            bool down;
            int threshold = match->input.axis.axis_min + (match->input.axis.axis_max - match->input.axis.axis_min) / 2;
            if (match->input.axis.axis_max < match->input.axis.axis_min) {
                down = (value <= threshold);
            } else {
                down = (value >= threshold);
            }
            SDL_SendGamepadButton(timestamp, gamepad, match->output.button, down);
        }
    }
    gamepad->last_match_axis[axis] = match;
}

static void HandleJoystickButton(Uint64 timestamp, SDL_Gamepad *gamepad, int button, bool down)
{
    int i;

    SDL_AssertJoysticksLocked();

    for (i = 0; i < gamepad->num_bindings; ++i) {
        SDL_GamepadBinding *binding = &gamepad->bindings[i];
        if (binding->input_type == SDL_GAMEPAD_BINDTYPE_BUTTON &&
            button == binding->input.button) {
            if (binding->output_type == SDL_GAMEPAD_BINDTYPE_AXIS) {
                int value = down ? binding->output.axis.axis_max : binding->output.axis.axis_min;
                SDL_SendGamepadAxis(timestamp, gamepad, binding->output.axis.axis, (Sint16)value);
            } else {
                SDL_SendGamepadButton(timestamp, gamepad, binding->output.button, down);
            }
            break;
        }
    }
}

static void HandleJoystickHat(Uint64 timestamp, SDL_Gamepad *gamepad, int hat, Uint8 value)
{
    int i;
    Uint8 last_mask, changed_mask;

    SDL_AssertJoysticksLocked();

    last_mask = gamepad->last_hat_mask[hat];
    changed_mask = (last_mask ^ value);
    for (i = 0; i < gamepad->num_bindings; ++i) {
        SDL_GamepadBinding *binding = &gamepad->bindings[i];
        if (binding->input_type == SDL_GAMEPAD_BINDTYPE_HAT && hat == binding->input.hat.hat) {
            if ((changed_mask & binding->input.hat.hat_mask) != 0) {
                if (value & binding->input.hat.hat_mask) {
                    if (binding->output_type == SDL_GAMEPAD_BINDTYPE_AXIS) {
                        SDL_SendGamepadAxis(timestamp, gamepad, binding->output.axis.axis, (Sint16)binding->output.axis.axis_max);
                    } else {
                        SDL_SendGamepadButton(timestamp, gamepad, binding->output.button, true);
                    }
                } else {
                    ResetOutput(timestamp, gamepad, binding);
                }
            }
        }
    }
    gamepad->last_hat_mask[hat] = value;
}

/* The joystick layer will _also_ send events to recenter before disconnect,
    but it has to make (sometimes incorrect) guesses at what being "centered"
    is. The gamepad layer, however, can set a definite logical idle
    position, so set them all here. If we happened to already be at the
    center thanks to the joystick layer or idle hands, this won't generate
    duplicate events. */
static void RecenterGamepad(SDL_Gamepad *gamepad)
{
    int i;
    Uint64 timestamp = SDL_GetTicksNS();

    for (i = 0; i < SDL_GAMEPAD_BUTTON_COUNT; ++i) {
        SDL_GamepadButton button = (SDL_GamepadButton)i;
        if (SDL_GetGamepadButton(gamepad, button)) {
            SDL_SendGamepadButton(timestamp, gamepad, button, false);
        }
    }

    for (i = 0; i < SDL_GAMEPAD_AXIS_COUNT; ++i) {
        SDL_GamepadAxis axis = (SDL_GamepadAxis)i;
        if (SDL_GetGamepadAxis(gamepad, axis) != 0) {
            SDL_SendGamepadAxis(timestamp, gamepad, axis, 0);
        }
    }
}

void SDL_PrivateGamepadAdded(SDL_JoystickID instance_id)
{
    SDL_Event event;

    if (!SDL_gamepads_initialized || SDL_IsJoystickBeingAdded()) {
        return;
    }

    event.type = SDL_EVENT_GAMEPAD_ADDED;
    event.common.timestamp = 0;
    event.gdevice.which = instance_id;
    SDL_PushEvent(&event);
}

void SDL_PrivateGamepadRemoved(SDL_JoystickID instance_id)
{
    SDL_Event event;
    SDL_Gamepad *gamepad;

    SDL_AssertJoysticksLocked();

    if (!SDL_gamepads_initialized) {
        return;
    }

    for (gamepad = SDL_gamepads; gamepad; gamepad = gamepad->next) {
        if (gamepad->joystick->instance_id == instance_id) {
            RecenterGamepad(gamepad);
            break;
        }
    }

    event.type = SDL_EVENT_GAMEPAD_REMOVED;
    event.common.timestamp = 0;
    event.gdevice.which = instance_id;
    SDL_PushEvent(&event);
}

static void SDL_PrivateGamepadRemapped(SDL_JoystickID instance_id)
{
    SDL_Event event;

    if (!SDL_gamepads_initialized || SDL_IsJoystickBeingAdded()) {
        return;
    }

    event.type = SDL_EVENT_GAMEPAD_REMAPPED;
    event.common.timestamp = 0;
    event.gdevice.which = instance_id;
    SDL_PushEvent(&event);
}

/*
 * Event filter to fire gamepad events from joystick ones
 */
static bool SDLCALL SDL_GamepadEventWatcher(void *userdata, SDL_Event *event)
{
    SDL_Gamepad *gamepad;

    switch (event->type) {
    case SDL_EVENT_JOYSTICK_AXIS_MOTION:
    {
        SDL_AssertJoysticksLocked();

        for (gamepad = SDL_gamepads; gamepad; gamepad = gamepad->next) {
            if (gamepad->joystick->instance_id == event->jaxis.which) {
                HandleJoystickAxis(event->common.timestamp, gamepad, event->jaxis.axis, event->jaxis.value);
                break;
            }
        }
    } break;
    case SDL_EVENT_JOYSTICK_BUTTON_DOWN:
    case SDL_EVENT_JOYSTICK_BUTTON_UP:
    {
        SDL_AssertJoysticksLocked();

        for (gamepad = SDL_gamepads; gamepad; gamepad = gamepad->next) {
            if (gamepad->joystick->instance_id == event->jbutton.which) {
                HandleJoystickButton(event->common.timestamp, gamepad, event->jbutton.button, event->jbutton.down);
                break;
            }
        }
    } break;
    case SDL_EVENT_JOYSTICK_HAT_MOTION:
    {
        SDL_AssertJoysticksLocked();

        for (gamepad = SDL_gamepads; gamepad; gamepad = gamepad->next) {
            if (gamepad->joystick->instance_id == event->jhat.which) {
                HandleJoystickHat(event->common.timestamp, gamepad, event->jhat.hat, event->jhat.value);
                break;
            }
        }
    } break;
    case SDL_EVENT_JOYSTICK_UPDATE_COMPLETE:
    {
        SDL_AssertJoysticksLocked();

        if (SDL_EventEnabled(SDL_EVENT_GAMEPAD_UPDATE_COMPLETE)) {
            for (gamepad = SDL_gamepads; gamepad; gamepad = gamepad->next) {
                if (gamepad->joystick->instance_id == event->jdevice.which) {
                    SDL_Event deviceevent;

                    deviceevent.type = SDL_EVENT_GAMEPAD_UPDATE_COMPLETE;
                    deviceevent.common.timestamp = event->jdevice.timestamp;
                    deviceevent.gdevice.which = event->jdevice.which;
                    SDL_PushEvent(&deviceevent);
                    break;
                }
            }
        }
    } break;
    default:
        break;
    }

    return true;
}

/* SDL defines sensor orientation relative to the device natural
   orientation, so when it's changed orientation to be used as a
   gamepad, change the sensor orientation to match.
 */
static void AdjustSensorOrientation(const SDL_Joystick *joystick, const float *src, float *dst)
{
    unsigned int i, j;

    SDL_AssertJoysticksLocked();

    for (i = 0; i < 3; ++i) {
        dst[i] = 0.0f;
        for (j = 0; j < 3; ++j) {
            dst[i] += joystick->sensor_transform[i][j] * src[j];
        }
    }
}

/*
 * Event filter to fire gamepad sensor events from system sensor events
 *
 * We don't use SDL_GamepadEventWatcher() for this because we want to
 * deliver gamepad sensor events when system sensor events are disabled,
 * and we also need to avoid a potential deadlock where joystick event
 * delivery locks the joysticks and then the event queue, but sensor
 * event delivery would lock the event queue and then from within the
 * event watcher function lock the joysticks.
 */
void SDL_GamepadSensorWatcher(Uint64 timestamp, SDL_SensorID sensor, Uint64 sensor_timestamp, float *data, int num_values)
{
    SDL_Gamepad *gamepad;

    SDL_LockJoysticks();
    for (gamepad = SDL_gamepads; gamepad; gamepad = gamepad->next) {
        if (gamepad->joystick->accel && gamepad->joystick->accel_sensor == sensor) {
            float gamepad_data[3];
            AdjustSensorOrientation(gamepad->joystick, data, gamepad_data);
            SDL_SendJoystickSensor(timestamp, gamepad->joystick, SDL_SENSOR_ACCEL, sensor_timestamp, gamepad_data, SDL_arraysize(gamepad_data));
        }
        if (gamepad->joystick->gyro && gamepad->joystick->gyro_sensor == sensor) {
            float gamepad_data[3];
            AdjustSensorOrientation(gamepad->joystick, data, gamepad_data);
            SDL_SendJoystickSensor(timestamp, gamepad->joystick, SDL_SENSOR_GYRO, sensor_timestamp, gamepad_data, SDL_arraysize(gamepad_data));
        }
    }
    SDL_UnlockJoysticks();
}

static void PushMappingChangeTracking(void)
{
    MappingChangeTracker *tracker;
    int i, num_joysticks;

    SDL_AssertJoysticksLocked();

    if (s_mappingChangeTracker) {
        ++s_mappingChangeTracker->refcount;
        return;
    }
    s_mappingChangeTracker = (MappingChangeTracker *)SDL_calloc(1, sizeof(*tracker));
    s_mappingChangeTracker->refcount = 1;

    // Save the list of joysticks and associated mappings
    tracker = s_mappingChangeTracker;
    tracker->joysticks = SDL_GetJoysticks(&num_joysticks);
    if (!tracker->joysticks) {
        return;
    }
    if (num_joysticks == 0) {
        return;
    }
    tracker->joystick_mappings = (GamepadMapping_t **)SDL_malloc(num_joysticks * sizeof(*tracker->joystick_mappings));
    if (!tracker->joystick_mappings) {
        return;
    }
    for (i = 0; i < num_joysticks; ++i) {
        tracker->joystick_mappings[i] = SDL_PrivateGetGamepadMapping(tracker->joysticks[i], false);
    }
}

static void AddMappingChangeTracking(GamepadMapping_t *mapping)
{
    MappingChangeTracker *tracker;
    int num_mappings;
    GamepadMapping_t **new_mappings;

    SDL_AssertJoysticksLocked();

    SDL_assert(s_mappingChangeTracker != NULL);
    tracker = s_mappingChangeTracker;
    num_mappings = tracker->num_changed_mappings;
    new_mappings = (GamepadMapping_t **)SDL_realloc(tracker->changed_mappings, (num_mappings + 1) * sizeof(*new_mappings));
    if (new_mappings) {
        tracker->changed_mappings = new_mappings;
        tracker->changed_mappings[num_mappings] = mapping;
        tracker->num_changed_mappings = (num_mappings + 1);
    }
}

static bool HasMappingChangeTracking(MappingChangeTracker *tracker, GamepadMapping_t *mapping)
{
    int i;

    SDL_AssertJoysticksLocked();

    for (i = 0; i < tracker->num_changed_mappings; ++i) {
        if (tracker->changed_mappings[i] == mapping) {
            return true;
        }
    }
    return false;
}

static void PopMappingChangeTracking(void)
{
    int i;
    MappingChangeTracker *tracker;

    SDL_AssertJoysticksLocked();

    SDL_assert(s_mappingChangeTracker != NULL);
    tracker = s_mappingChangeTracker;
    --tracker->refcount;
    if (tracker->refcount > 0) {
        return;
    }
    s_mappingChangeTracker = NULL;

    // Now check to see what gamepads changed because of the mapping changes
    if (tracker->joysticks && tracker->joystick_mappings) {
        for (i = 0; tracker->joysticks[i]; ++i) {
            // Looking up the new mapping might create one and associate it with the gamepad (and generate events)
            SDL_JoystickID joystick = tracker->joysticks[i];
            SDL_Gamepad *gamepad = SDL_GetGamepadFromID(joystick);
            GamepadMapping_t *new_mapping = SDL_PrivateGetGamepadMapping(joystick, false);
            GamepadMapping_t *old_mapping = gamepad ? gamepad->mapping : tracker->joystick_mappings[i];

            if (new_mapping && !old_mapping) {
                SDL_InsertIntoHashTable(s_gamepadInstanceIDs, (void *)(uintptr_t)joystick, (const void *)true, true);
                SDL_PrivateGamepadAdded(joystick);
            } else if (old_mapping && !new_mapping) {
                SDL_InsertIntoHashTable(s_gamepadInstanceIDs, (void *)(uintptr_t)joystick, (const void *)false, true);
                SDL_PrivateGamepadRemoved(joystick);
            } else if (old_mapping != new_mapping || HasMappingChangeTracking(tracker, new_mapping)) {
                if (gamepad) {
                    SDL_PrivateLoadButtonMapping(gamepad, new_mapping);
                }
                SDL_PrivateGamepadRemapped(joystick);
            }
        }
    }

    SDL_free(tracker->joysticks);
    SDL_free(tracker->joystick_mappings);
    SDL_free(tracker->changed_mappings);
    SDL_free(tracker);
}

#ifdef SDL_PLATFORM_ANDROID
/*
 * Helper function to guess at a mapping based on the elements reported for this gamepad
 */
static GamepadMapping_t *SDL_CreateMappingForAndroidGamepad(SDL_GUID guid)
{
    const int face_button_mask = ((1 << SDL_GAMEPAD_BUTTON_SOUTH) |
                                  (1 << SDL_GAMEPAD_BUTTON_EAST) |
                                  (1 << SDL_GAMEPAD_BUTTON_WEST) |
                                  (1 << SDL_GAMEPAD_BUTTON_NORTH));
    bool existing;
    char mapping_string[1024];
    int button_mask;
    int axis_mask;

    button_mask = SDL_Swap16LE(*(Uint16 *)(&guid.data[sizeof(guid.data) - 4]));
    axis_mask = SDL_Swap16LE(*(Uint16 *)(&guid.data[sizeof(guid.data) - 2]));
    if (!button_mask && !axis_mask) {
        // Accelerometer, shouldn't have a gamepad mapping
        return NULL;
    }
    if (!(button_mask & face_button_mask)) {
        // We don't know what buttons or axes are supported, don't make up a mapping
        return NULL;
    }

    SDL_strlcpy(mapping_string, "none,*,", sizeof(mapping_string));

    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_SOUTH)) {
        SDL_strlcat(mapping_string, "a:b0,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_EAST)) {
        SDL_strlcat(mapping_string, "b:b1,", sizeof(mapping_string));
    } else if (button_mask & (1 << SDL_GAMEPAD_BUTTON_BACK)) {
        // Use the back button as "B" for easy UI navigation with TV remotes
        SDL_strlcat(mapping_string, "b:b4,", sizeof(mapping_string));
        button_mask &= ~(1 << SDL_GAMEPAD_BUTTON_BACK);
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_WEST)) {
        SDL_strlcat(mapping_string, "x:b2,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_NORTH)) {
        SDL_strlcat(mapping_string, "y:b3,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_BACK)) {
        SDL_strlcat(mapping_string, "back:b4,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_GUIDE)) {
        // The guide button generally isn't functional (or acts as a home button) on most Android gamepads before Android 11
        if (SDL_GetAndroidSDKVersion() >= 30 /* Android 11 */) {
            SDL_strlcat(mapping_string, "guide:b5,", sizeof(mapping_string));
        }
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_START)) {
        SDL_strlcat(mapping_string, "start:b6,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_LEFT_STICK)) {
        SDL_strlcat(mapping_string, "leftstick:b7,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_RIGHT_STICK)) {
        SDL_strlcat(mapping_string, "rightstick:b8,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_LEFT_SHOULDER)) {
        SDL_strlcat(mapping_string, "leftshoulder:b9,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER)) {
        SDL_strlcat(mapping_string, "rightshoulder:b10,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_DPAD_UP)) {
        SDL_strlcat(mapping_string, "dpup:b11,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_DPAD_DOWN)) {
        SDL_strlcat(mapping_string, "dpdown:b12,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_DPAD_LEFT)) {
        SDL_strlcat(mapping_string, "dpleft:b13,", sizeof(mapping_string));
    }
    if (button_mask & (1 << SDL_GAMEPAD_BUTTON_DPAD_RIGHT)) {
        SDL_strlcat(mapping_string, "dpright:b14,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_GAMEPAD_AXIS_LEFTX)) {
        SDL_strlcat(mapping_string, "leftx:a0,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_GAMEPAD_AXIS_LEFTY)) {
        SDL_strlcat(mapping_string, "lefty:a1,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_GAMEPAD_AXIS_RIGHTX)) {
        SDL_strlcat(mapping_string, "rightx:a2,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_GAMEPAD_AXIS_RIGHTY)) {
        SDL_strlcat(mapping_string, "righty:a3,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_GAMEPAD_AXIS_LEFT_TRIGGER)) {
        SDL_strlcat(mapping_string, "lefttrigger:a4,", sizeof(mapping_string));
    }
    if (axis_mask & (1 << SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)) {
        SDL_strlcat(mapping_string, "righttrigger:a5,", sizeof(mapping_string));
    }

    return SDL_PrivateAddMappingForGUID(guid, mapping_string, &existing, SDL_GAMEPAD_MAPPING_PRIORITY_DEFAULT);
}
#endif // SDL_PLATFORM_ANDROID

/*
 * Helper function to guess at a mapping for HIDAPI gamepads
 */
static GamepadMapping_t *SDL_CreateMappingForHIDAPIGamepad(SDL_GUID guid)
{
    bool existing;
    char mapping_string[1024];
    Uint16 vendor;
    Uint16 product;

    SDL_strlcpy(mapping_string, "none,*,", sizeof(mapping_string));

    SDL_GetJoystickGUIDInfo(guid, &vendor, &product, NULL, NULL);

    if ((vendor == USB_VENDOR_NINTENDO && product == USB_PRODUCT_NINTENDO_GAMECUBE_ADAPTER) ||
        (vendor == USB_VENDOR_DRAGONRISE &&
         (product == USB_PRODUCT_EVORETRO_GAMECUBE_ADAPTER1 ||
          product == USB_PRODUCT_EVORETRO_GAMECUBE_ADAPTER2))) {
        // GameCube driver has 12 buttons and 6 axes
        SDL_strlcat(mapping_string, "a:b0,b:b1,dpdown:b6,dpleft:b4,dpright:b5,dpup:b7,lefttrigger:a4,leftx:a0,lefty:a1~,rightshoulder:b9,righttrigger:a5,rightx:a2,righty:a3~,start:b8,x:b2,y:b3,", sizeof(mapping_string));
    } else if (vendor == USB_VENDOR_NINTENDO &&
               (guid.data[15] == k_eSwitchDeviceInfoControllerType_HVCLeft ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_HVCRight ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_NESLeft ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_NESRight ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_SNES ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_N64 ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_SEGA_Genesis ||
                guid.data[15] == k_eWiiExtensionControllerType_None ||
                guid.data[15] == k_eWiiExtensionControllerType_Nunchuk ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_JoyConLeft ||
                guid.data[15] == k_eSwitchDeviceInfoControllerType_JoyConRight)) {
        switch (guid.data[15]) {
        case k_eSwitchDeviceInfoControllerType_HVCLeft:
            SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b9,rightshoulder:b10,start:b6,", sizeof(mapping_string));
            break;
        case k_eSwitchDeviceInfoControllerType_HVCRight:
            SDL_strlcat(mapping_string, "a:b0,b:b1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b9,rightshoulder:b10,", sizeof(mapping_string));
            break;
        case k_eSwitchDeviceInfoControllerType_NESLeft:
        case k_eSwitchDeviceInfoControllerType_NESRight:
            SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b9,rightshoulder:b10,start:b6,", sizeof(mapping_string));
            break;
        case k_eSwitchDeviceInfoControllerType_SNES:
            SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b9,lefttrigger:a4,rightshoulder:b10,righttrigger:a5,start:b6,x:b2,y:b3,hint:!SDL_GAMECONTROLLER_USE_BUTTON_LABELS:=1,", sizeof(mapping_string));
            break;
        case k_eSwitchDeviceInfoControllerType_N64:
            SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b5,leftshoulder:b9,leftstick:b7,lefttrigger:a4,leftx:a0,lefty:a1,rightshoulder:b10,righttrigger:a5,start:b6,x:b2,y:b3,misc1:b11,", sizeof(mapping_string));
            break;
        case k_eSwitchDeviceInfoControllerType_SEGA_Genesis:
            SDL_strlcat(mapping_string, "a:b0,b:b1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b5,leftshoulder:b9,rightshoulder:b10,righttrigger:a5,start:b6,x:b2,y:b3,misc1:b11,", sizeof(mapping_string));
            break;
        case k_eWiiExtensionControllerType_None:
            SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b5,start:b6,x:b2,y:b3,", sizeof(mapping_string));
            break;
        case k_eWiiExtensionControllerType_Nunchuk:
        {
            // FIXME: Should we map this to the left or right side?
            const bool map_nunchuck_left_side = true;

            if (map_nunchuck_left_side) {
                SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b5,leftshoulder:b9,lefttrigger:a4,leftx:a0,lefty:a1,start:b6,x:b2,y:b3,", sizeof(mapping_string));
            } else {
                SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:b12,dpleft:b13,dpright:b14,dpup:b11,guide:b5,rightshoulder:b9,righttrigger:a4,rightx:a0,righty:a1,start:b6,x:b2,y:b3,", sizeof(mapping_string));
            }
        } break;
        default:
            if (SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_VERTICAL_JOY_CONS, false)) {
                // Vertical mode
                if (guid.data[15] == k_eSwitchDeviceInfoControllerType_JoyConLeft) {
                    SDL_strlcat(mapping_string, "back:b4,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,leftshoulder:b9,leftstick:b7,lefttrigger:a4,leftx:a0,lefty:a1,misc1:b11,paddle2:b13,paddle4:b15,", sizeof(mapping_string));
                } else {
                    SDL_strlcat(mapping_string, "a:b0,b:b1,guide:b5,rightshoulder:b10,rightstick:b8,righttrigger:a5,rightx:a2,righty:a3,start:b6,x:b2,y:b3,paddle1:b12,paddle3:b14,", sizeof(mapping_string));
                }
            } else {
                // Mini gamepad mode
                if (guid.data[15] == k_eSwitchDeviceInfoControllerType_JoyConLeft) {
                    SDL_strlcat(mapping_string, "a:b0,b:b1,guide:b5,leftshoulder:b9,leftstick:b7,leftx:a0,lefty:a1,rightshoulder:b10,start:b6,x:b2,y:b3,paddle2:b13,paddle4:b15,", sizeof(mapping_string));
                } else {
                    SDL_strlcat(mapping_string, "a:b0,b:b1,guide:b5,leftshoulder:b9,leftstick:b7,leftx:a0,lefty:a1,rightshoulder:b10,start:b6,x:b2,y:b3,paddle1:b12,paddle3:b14,", sizeof(mapping_string));
                }
            }
            break;
        }
    } else {
        // All other gamepads have the standard set of 19 buttons and 6 axes
        SDL_strlcat(mapping_string, "a:b0,b:b1,back:b4,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b5,leftshoulder:b9,leftstick:b7,lefttrigger:a4,leftx:a0,lefty:a1,rightshoulder:b10,rightstick:b8,righttrigger:a5,rightx:a2,righty:a3,start:b6,x:b2,y:b3,", sizeof(mapping_string));

        if (SDL_IsJoystickSteamController(vendor, product)) {
            // Steam controllers have 2 back paddle buttons
            SDL_strlcat(mapping_string, "paddle1:b11,paddle2:b12,", sizeof(mapping_string));
        } else if (SDL_IsJoystickNintendoSwitchPro(vendor, product) ||
                   SDL_IsJoystickNintendoSwitchProInputOnly(vendor, product)) {
            // Nintendo Switch Pro controllers have a screenshot button
            SDL_strlcat(mapping_string, "misc1:b11,", sizeof(mapping_string));
        } else if (SDL_IsJoystickNintendoSwitchJoyConPair(vendor, product)) {
            // The Nintendo Switch Joy-Con combined controllers has a share button and paddles
            SDL_strlcat(mapping_string, "misc1:b11,paddle1:b12,paddle2:b13,paddle3:b14,paddle4:b15,", sizeof(mapping_string));
        } else if (SDL_IsJoystickAmazonLunaController(vendor, product)) {
            // Amazon Luna Controller has a mic button under the guide button
            SDL_strlcat(mapping_string, "misc1:b11,", sizeof(mapping_string));
        } else if (SDL_IsJoystickGoogleStadiaController(vendor, product)) {
            // The Google Stadia controller has a share button and a Google Assistant button
            SDL_strlcat(mapping_string, "misc1:b11,misc2:b12", sizeof(mapping_string));
        } else if (SDL_IsJoystickNVIDIASHIELDController(vendor, product)) {
            // The NVIDIA SHIELD controller has a share button between back and start buttons
            SDL_strlcat(mapping_string, "misc1:b11,", sizeof(mapping_string));

            if (product == USB_PRODUCT_NVIDIA_SHIELD_CONTROLLER_V103) {
                // The original SHIELD controller has a touchpad and plus/minus buttons as well
                SDL_strlcat(mapping_string, "touchpad:b12,misc2:b13,misc3:b14", sizeof(mapping_string));
            }
        } else if (SDL_IsJoystickHoriSteamController(vendor, product)) {
            /* The Wireless HORIPad for Steam has QAM, Steam, Capsense L/R Sticks, 2 rear buttons, and 2 misc buttons */
            SDL_strlcat(mapping_string, "paddle1:b13,paddle2:b12,paddle3:b15,paddle4:b14,misc2:b11,misc3:b16,misc4:b17", sizeof(mapping_string));
        } else {
            switch (SDL_GetGamepadTypeFromGUID(guid, NULL)) {
            case SDL_GAMEPAD_TYPE_PS4:
                // PS4 controllers have an additional touchpad button
                SDL_strlcat(mapping_string, "touchpad:b11,", sizeof(mapping_string));
                break;
            case SDL_GAMEPAD_TYPE_PS5:
                // PS5 controllers have a microphone button and an additional touchpad button
                SDL_strlcat(mapping_string, "touchpad:b11,misc1:b12,", sizeof(mapping_string));
                // DualSense Edge controllers have paddles
                if (SDL_IsJoystickDualSenseEdge(vendor, product)) {
                    SDL_strlcat(mapping_string, "paddle1:b16,paddle2:b15,paddle3:b14,paddle4:b13,", sizeof(mapping_string));
                }
                break;
            case SDL_GAMEPAD_TYPE_XBOXONE:
                if (SDL_IsJoystickXboxOneElite(vendor, product)) {
                    // XBox One Elite Controllers have 4 back paddle buttons
                    SDL_strlcat(mapping_string, "paddle1:b11,paddle2:b13,paddle3:b12,paddle4:b14,", sizeof(mapping_string));
                } else if (SDL_IsJoystickXboxSeriesX(vendor, product)) {
                    // XBox Series X Controllers have a share button under the guide button
                    SDL_strlcat(mapping_string, "misc1:b11,", sizeof(mapping_string));
                }
                break;
            default:
                if (vendor == 0 && product == 0) {
                    // This is a Bluetooth Nintendo Switch Pro controller
                    SDL_strlcat(mapping_string, "misc1:b11,", sizeof(mapping_string));
                }
                break;
            }
        }
    }

    return SDL_PrivateAddMappingForGUID(guid, mapping_string, &existing, SDL_GAMEPAD_MAPPING_PRIORITY_DEFAULT);
}

/*
 * Helper function to guess at a mapping for RAWINPUT gamepads
 */
static GamepadMapping_t *SDL_CreateMappingForRAWINPUTGamepad(SDL_GUID guid)
{
    bool existing;
    char mapping_string[1024];

    SDL_strlcpy(mapping_string, "none,*,", sizeof(mapping_string));
    SDL_strlcat(mapping_string, "a:b0,b:b1,x:b2,y:b3,back:b6,guide:b10,start:b7,leftstick:b8,rightstick:b9,leftshoulder:b4,rightshoulder:b5,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:a4,righttrigger:a5,", sizeof(mapping_string));

    return SDL_PrivateAddMappingForGUID(guid, mapping_string, &existing, SDL_GAMEPAD_MAPPING_PRIORITY_DEFAULT);
}

/*
 * Helper function to guess at a mapping for WGI gamepads
 */
static GamepadMapping_t *SDL_CreateMappingForWGIGamepad(SDL_GUID guid)
{
    bool existing;
    char mapping_string[1024];

    if (guid.data[15] != SDL_JOYSTICK_TYPE_GAMEPAD) {
        return NULL;
    }

    SDL_strlcpy(mapping_string, "none,*,", sizeof(mapping_string));
    SDL_strlcat(mapping_string, "a:b0,b:b1,x:b2,y:b3,back:b6,start:b7,leftstick:b8,rightstick:b9,leftshoulder:b4,rightshoulder:b5,dpup:b10,dpdown:b12,dpleft:b13,dpright:b11,leftx:a1,lefty:a0~,rightx:a3,righty:a2~,lefttrigger:a4,righttrigger:a5,", sizeof(mapping_string));

    return SDL_PrivateAddMappingForGUID(guid, mapping_string, &existing, SDL_GAMEPAD_MAPPING_PRIORITY_DEFAULT);
}

/*
 * Helper function to scan the mappings database for a gamepad with the specified GUID
 */
static GamepadMapping_t *SDL_PrivateMatchGamepadMappingForGUID(SDL_GUID guid, bool match_version, bool exact_match_crc)
{
    GamepadMapping_t *mapping, *best_match = NULL;
    Uint16 crc = 0;

    SDL_AssertJoysticksLocked();

    SDL_GetJoystickGUIDInfo(guid, NULL, NULL, NULL, &crc);

    // Clear the CRC from the GUID for matching, the mappings never include it in the GUID
    SDL_SetJoystickGUIDCRC(&guid, 0);

    if (!match_version) {
        SDL_SetJoystickGUIDVersion(&guid, 0);
    }

    for (mapping = s_pSupportedGamepads; mapping; mapping = mapping->next) {
        SDL_GUID mapping_guid;

        if (SDL_memcmp(&mapping->guid, &s_zeroGUID, sizeof(mapping->guid)) == 0) {
            continue;
        }

        SDL_memcpy(&mapping_guid, &mapping->guid, sizeof(mapping_guid));
        if (!match_version) {
            SDL_SetJoystickGUIDVersion(&mapping_guid, 0);
        }

        if (SDL_memcmp(&guid, &mapping_guid, sizeof(guid)) == 0) {
            const char *crc_string = SDL_strstr(mapping->mapping, SDL_GAMEPAD_CRC_FIELD);
            if (crc_string) {
                Uint16 mapping_crc = (Uint16)SDL_strtol(crc_string + SDL_GAMEPAD_CRC_FIELD_SIZE, NULL, 16);
                if (mapping_crc != crc) {
                    // This mapping specified a CRC and they don't match
                    continue;
                }

                // An exact match, including CRC
                return mapping;
            } else if (crc && exact_match_crc) {
                continue;
            }

            if (!best_match) {
                best_match = mapping;
            }
        }
    }
    return best_match;
}

/*
 * Helper function to scan the mappings database for a gamepad with the specified GUID
 */
static GamepadMapping_t *SDL_PrivateGetGamepadMappingForGUID(SDL_GUID guid, bool adding_mapping)
{
    GamepadMapping_t *mapping;

    mapping = SDL_PrivateMatchGamepadMappingForGUID(guid, true, adding_mapping);
    if (mapping) {
        return mapping;
    }

    if (adding_mapping) {
        // We didn't find an existing mapping
        return NULL;
    }

    // Try harder to get the best match, or create a mapping

    if (SDL_JoystickGUIDUsesVersion(guid)) {
        // Try again, ignoring the version
        mapping = SDL_PrivateMatchGamepadMappingForGUID(guid, false, false);
        if (mapping) {
            return mapping;
        }
    }

#ifdef SDL_JOYSTICK_XINPUT
    if (SDL_IsJoystickXInput(guid)) {
        // This is an XInput device
        return s_pXInputMapping;
    }
#endif
    if (SDL_IsJoystickHIDAPI(guid)) {
        mapping = SDL_CreateMappingForHIDAPIGamepad(guid);
    } else if (SDL_IsJoystickRAWINPUT(guid)) {
        mapping = SDL_CreateMappingForRAWINPUTGamepad(guid);
    } else if (SDL_IsJoystickWGI(guid)) {
        mapping = SDL_CreateMappingForWGIGamepad(guid);
    } else if (SDL_IsJoystickVIRTUAL(guid)) {
        // We'll pick up a robust mapping in VIRTUAL_JoystickGetGamepadMapping
#ifdef SDL_PLATFORM_ANDROID
    } else {
        mapping = SDL_CreateMappingForAndroidGamepad(guid);
#endif
    }
    return mapping;
}

static const char *map_StringForGamepadType[] = {
    "unknown",
    "standard",
    "xbox360",
    "xboxone",
    "ps3",
    "ps4",
    "ps5",
    "switchpro",
    "joyconleft",
    "joyconright",
    "joyconpair"
};
SDL_COMPILE_TIME_ASSERT(map_StringForGamepadType, SDL_arraysize(map_StringForGamepadType) == SDL_GAMEPAD_TYPE_COUNT);

/*
 * convert a string to its enum equivalent
 */
SDL_GamepadType SDL_GetGamepadTypeFromString(const char *str)
{
    int i;

    if (!str || str[0] == '\0') {
        return SDL_GAMEPAD_TYPE_UNKNOWN;
    }

    if (*str == '+' || *str == '-') {
        ++str;
    }

    for (i = 0; i < SDL_arraysize(map_StringForGamepadType); ++i) {
        if (SDL_strcasecmp(str, map_StringForGamepadType[i]) == 0) {
            return (SDL_GamepadType)i;
        }
    }
    return SDL_GAMEPAD_TYPE_UNKNOWN;
}

/*
 * convert an enum to its string equivalent
 */
const char *SDL_GetGamepadStringForType(SDL_GamepadType type)
{
    if (type >= SDL_GAMEPAD_TYPE_STANDARD && type < SDL_GAMEPAD_TYPE_COUNT) {
        return map_StringForGamepadType[type];
    }
    return NULL;
}

static const char *map_StringForGamepadAxis[] = {
    "leftx",
    "lefty",
    "rightx",
    "righty",
    "lefttrigger",
    "righttrigger"
};
SDL_COMPILE_TIME_ASSERT(map_StringForGamepadAxis, SDL_arraysize(map_StringForGamepadAxis) == SDL_GAMEPAD_AXIS_COUNT);

/*
 * convert a string to its enum equivalent
 */
SDL_GamepadAxis SDL_GetGamepadAxisFromString(const char *str)
{
    int i;

    if (!str || str[0] == '\0') {
        return SDL_GAMEPAD_AXIS_INVALID;
    }

    if (*str == '+' || *str == '-') {
        ++str;
    }

    for (i = 0; i < SDL_arraysize(map_StringForGamepadAxis); ++i) {
        if (SDL_strcasecmp(str, map_StringForGamepadAxis[i]) == 0) {
            return (SDL_GamepadAxis)i;
        }
    }
    return SDL_GAMEPAD_AXIS_INVALID;
}

/*
 * convert an enum to its string equivalent
 */
const char *SDL_GetGamepadStringForAxis(SDL_GamepadAxis axis)
{
    if (axis > SDL_GAMEPAD_AXIS_INVALID && axis < SDL_GAMEPAD_AXIS_COUNT) {
        return map_StringForGamepadAxis[axis];
    }
    return NULL;
}

static const char *map_StringForGamepadButton[] = {
    "a",
    "b",
    "x",
    "y",
    "back",
    "guide",
    "start",
    "leftstick",
    "rightstick",
    "leftshoulder",
    "rightshoulder",
    "dpup",
    "dpdown",
    "dpleft",
    "dpright",
    "misc1",
    "paddle1",
    "paddle2",
    "paddle3",
    "paddle4",
    "touchpad",
    "misc2",
    "misc3",
    "misc4",
    "misc5",
    "misc6"
};
SDL_COMPILE_TIME_ASSERT(map_StringForGamepadButton, SDL_arraysize(map_StringForGamepadButton) == SDL_GAMEPAD_BUTTON_COUNT);

/*
 * convert a string to its enum equivalent
 */
static SDL_GamepadButton SDL_PrivateGetGamepadButtonFromString(const char *str, bool baxy)
{
    int i;

    if (!str || str[0] == '\0') {
        return SDL_GAMEPAD_BUTTON_INVALID;
    }

    for (i = 0; i < SDL_arraysize(map_StringForGamepadButton); ++i) {
        if (SDL_strcasecmp(str, map_StringForGamepadButton[i]) == 0) {
            if (baxy) {
                // Need to swap face buttons
                switch (i) {
                case SDL_GAMEPAD_BUTTON_SOUTH:
                    return SDL_GAMEPAD_BUTTON_EAST;
                case SDL_GAMEPAD_BUTTON_EAST:
                    return SDL_GAMEPAD_BUTTON_SOUTH;
                case SDL_GAMEPAD_BUTTON_WEST:
                    return SDL_GAMEPAD_BUTTON_NORTH;
                case SDL_GAMEPAD_BUTTON_NORTH:
                    return SDL_GAMEPAD_BUTTON_WEST;
                default:
                    break;
                }
            }
            return (SDL_GamepadButton)i;
        }
    }
    return SDL_GAMEPAD_BUTTON_INVALID;
}
SDL_GamepadButton SDL_GetGamepadButtonFromString(const char *str)
{
    return SDL_PrivateGetGamepadButtonFromString(str, false);
}

/*
 * convert an enum to its string equivalent
 */
const char *SDL_GetGamepadStringForButton(SDL_GamepadButton button)
{
    if (button > SDL_GAMEPAD_BUTTON_INVALID && button < SDL_GAMEPAD_BUTTON_COUNT) {
        return map_StringForGamepadButton[button];
    }
    return NULL;
}

/*
 * given a gamepad button name and a joystick name update our mapping structure with it
 */
static bool SDL_PrivateParseGamepadElement(SDL_Gamepad *gamepad, const char *szGameButton, const char *szJoystickButton)
{
    SDL_GamepadBinding bind;
    SDL_GamepadButton button;
    SDL_GamepadAxis axis;
    bool invert_input = false;
    char half_axis_input = 0;
    char half_axis_output = 0;
    int i;
    SDL_GamepadBinding *new_bindings;
    bool baxy_mapping = false;

    SDL_AssertJoysticksLocked();

    SDL_zero(bind);

    if (*szGameButton == '+' || *szGameButton == '-') {
        half_axis_output = *szGameButton++;
    }

    if (SDL_strstr(gamepad->mapping->mapping, ",hint:SDL_GAMECONTROLLER_USE_BUTTON_LABELS:=1") != NULL) {
        baxy_mapping = true;
    }

    axis = SDL_GetGamepadAxisFromString(szGameButton);
    button = SDL_PrivateGetGamepadButtonFromString(szGameButton, baxy_mapping);
    if (axis != SDL_GAMEPAD_AXIS_INVALID) {
        bind.output_type = SDL_GAMEPAD_BINDTYPE_AXIS;
        bind.output.axis.axis = axis;
        if (axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
            bind.output.axis.axis_min = 0;
            bind.output.axis.axis_max = SDL_JOYSTICK_AXIS_MAX;
        } else {
            if (half_axis_output == '+') {
                bind.output.axis.axis_min = 0;
                bind.output.axis.axis_max = SDL_JOYSTICK_AXIS_MAX;
            } else if (half_axis_output == '-') {
                bind.output.axis.axis_min = 0;
                bind.output.axis.axis_max = SDL_JOYSTICK_AXIS_MIN;
            } else {
                bind.output.axis.axis_min = SDL_JOYSTICK_AXIS_MIN;
                bind.output.axis.axis_max = SDL_JOYSTICK_AXIS_MAX;
            }
        }
    } else if (button != SDL_GAMEPAD_BUTTON_INVALID) {
        bind.output_type = SDL_GAMEPAD_BINDTYPE_BUTTON;
        bind.output.button = button;
    } else {
        return false;
    }

    if (*szJoystickButton == '+' || *szJoystickButton == '-') {
        half_axis_input = *szJoystickButton++;
    }
    if (szJoystickButton[SDL_strlen(szJoystickButton) - 1] == '~') {
        invert_input = true;
    }

    if (szJoystickButton[0] == 'a' && SDL_isdigit((unsigned char)szJoystickButton[1])) {
        bind.input_type = SDL_GAMEPAD_BINDTYPE_AXIS;
        bind.input.axis.axis = SDL_atoi(&szJoystickButton[1]);
        if (half_axis_input == '+') {
            bind.input.axis.axis_min = 0;
            bind.input.axis.axis_max = SDL_JOYSTICK_AXIS_MAX;
        } else if (half_axis_input == '-') {
            bind.input.axis.axis_min = 0;
            bind.input.axis.axis_max = SDL_JOYSTICK_AXIS_MIN;
        } else {
            bind.input.axis.axis_min = SDL_JOYSTICK_AXIS_MIN;
            bind.input.axis.axis_max = SDL_JOYSTICK_AXIS_MAX;
        }
        if (invert_input) {
            int tmp = bind.input.axis.axis_min;
            bind.input.axis.axis_min = bind.input.axis.axis_max;
            bind.input.axis.axis_max = tmp;
        }
    } else if (szJoystickButton[0] == 'b' && SDL_isdigit((unsigned char)szJoystickButton[1])) {
        bind.input_type = SDL_GAMEPAD_BINDTYPE_BUTTON;
        bind.input.button = SDL_atoi(&szJoystickButton[1]);
    } else if (szJoystickButton[0] == 'h' && SDL_isdigit((unsigned char)szJoystickButton[1]) &&
               szJoystickButton[2] == '.' && SDL_isdigit((unsigned char)szJoystickButton[3])) {
        int hat = SDL_atoi(&szJoystickButton[1]);
        int mask = SDL_atoi(&szJoystickButton[3]);
        bind.input_type = SDL_GAMEPAD_BINDTYPE_HAT;
        bind.input.hat.hat = hat;
        bind.input.hat.hat_mask = mask;
    } else {
        return false;
    }

    for (i = 0; i < gamepad->num_bindings; ++i) {
        if (SDL_memcmp(&gamepad->bindings[i], &bind, sizeof(bind)) == 0) {
            // We already have this binding, could be different face button names?
            return true;
        }
    }

    ++gamepad->num_bindings;
    new_bindings = (SDL_GamepadBinding *)SDL_realloc(gamepad->bindings, gamepad->num_bindings * sizeof(*gamepad->bindings));
    if (!new_bindings) {
        SDL_free(gamepad->bindings);
        gamepad->num_bindings = 0;
        gamepad->bindings = NULL;
        return false;
    }
    gamepad->bindings = new_bindings;
    gamepad->bindings[gamepad->num_bindings - 1] = bind;
    return true;
}

/*
 * given a gamepad mapping string update our mapping object
 */
static bool SDL_PrivateParseGamepadConfigString(SDL_Gamepad *gamepad, const char *pchString)
{
    char szGameButton[20];
    char szJoystickButton[20];
    bool bGameButton = true;
    int i = 0;
    const char *pchPos = pchString;

    SDL_zeroa(szGameButton);
    SDL_zeroa(szJoystickButton);

    while (pchPos && *pchPos) {
        if (*pchPos == ':') {
            i = 0;
            bGameButton = false;
        } else if (*pchPos == ' ') {

        } else if (*pchPos == ',') {
            i = 0;
            bGameButton = true;
            SDL_PrivateParseGamepadElement(gamepad, szGameButton, szJoystickButton);
            SDL_zeroa(szGameButton);
            SDL_zeroa(szJoystickButton);

        } else if (bGameButton) {
            if (i >= sizeof(szGameButton)) {
                szGameButton[sizeof(szGameButton) - 1] = '\0';
                return SDL_SetError("Button name too large: %s", szGameButton);
            }
            szGameButton[i] = *pchPos;
            i++;
        } else {
            if (i >= sizeof(szJoystickButton)) {
                szJoystickButton[sizeof(szJoystickButton) - 1] = '\0';
                return SDL_SetError("Joystick button name too large: %s", szJoystickButton);
            }
            szJoystickButton[i] = *pchPos;
            i++;
        }
        pchPos++;
    }

    // No more values if the string was terminated by a comma. Don't report an error.
    if (szGameButton[0] != '\0' || szJoystickButton[0] != '\0') {
        SDL_PrivateParseGamepadElement(gamepad, szGameButton, szJoystickButton);
    }
    return true;
}

static void SDL_UpdateGamepadType(SDL_Gamepad *gamepad)
{
    char *type_string, *comma;

    SDL_AssertJoysticksLocked();

    gamepad->type = SDL_GAMEPAD_TYPE_UNKNOWN;

    type_string = SDL_strstr(gamepad->mapping->mapping, SDL_GAMEPAD_TYPE_FIELD);
    if (type_string) {
        type_string += SDL_GAMEPAD_TYPE_FIELD_SIZE;
        comma = SDL_strchr(type_string, ',');
        if (comma) {
            *comma = '\0';
            gamepad->type = SDL_GetGamepadTypeFromString(type_string);
            *comma = ',';
        } else {
            gamepad->type = SDL_GetGamepadTypeFromString(type_string);
        }
    }
    if (gamepad->type == SDL_GAMEPAD_TYPE_UNKNOWN) {
        gamepad->type = SDL_GetRealGamepadTypeForID(gamepad->joystick->instance_id);
    }
}

static SDL_GamepadFaceStyle SDL_GetGamepadFaceStyleFromString(const char *string)
{
    if (SDL_strcmp(string, "abxy") == 0) {
        return SDL_GAMEPAD_FACE_STYLE_ABXY;
    } else if (SDL_strcmp(string, "bayx") == 0) {
        return SDL_GAMEPAD_FACE_STYLE_BAYX;
    } else if (SDL_strcmp(string, "sony") == 0) {
        return SDL_GAMEPAD_FACE_STYLE_SONY;
    } else {
        return SDL_GAMEPAD_FACE_STYLE_UNKNOWN;
    }
}

static SDL_GamepadFaceStyle SDL_GetGamepadFaceStyleForGamepadType(SDL_GamepadType type)
{
    switch (type) {
    case SDL_GAMEPAD_TYPE_PS3:
    case SDL_GAMEPAD_TYPE_PS4:
    case SDL_GAMEPAD_TYPE_PS5:
        return SDL_GAMEPAD_FACE_STYLE_SONY;
    case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO:
    case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT:
    case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT:
    case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR:
        return SDL_GAMEPAD_FACE_STYLE_BAYX;
    default:
        return SDL_GAMEPAD_FACE_STYLE_ABXY;
    }
}

static void SDL_UpdateGamepadFaceStyle(SDL_Gamepad *gamepad)
{
    char *face_string, *comma;

    SDL_AssertJoysticksLocked();

    gamepad->face_style = SDL_GAMEPAD_FACE_STYLE_UNKNOWN;

    face_string = SDL_strstr(gamepad->mapping->mapping, SDL_GAMEPAD_FACE_FIELD);
    if (face_string) {
        face_string += SDL_GAMEPAD_TYPE_FIELD_SIZE;
        comma = SDL_strchr(face_string, ',');
        if (comma) {
            *comma = '\0';
            gamepad->face_style = SDL_GetGamepadFaceStyleFromString(face_string);
            *comma = ',';
        } else {
            gamepad->face_style = SDL_GetGamepadFaceStyleFromString(face_string);
        }
    }

    if (gamepad->face_style == SDL_GAMEPAD_FACE_STYLE_UNKNOWN &&
        SDL_strstr(gamepad->mapping->mapping, "SDL_GAMECONTROLLER_USE_BUTTON_LABELS") != NULL) {
        // This controller uses Nintendo button style
        gamepad->face_style = SDL_GAMEPAD_FACE_STYLE_BAYX;
    }
    if (gamepad->face_style == SDL_GAMEPAD_FACE_STYLE_UNKNOWN) {
        gamepad->face_style = SDL_GetGamepadFaceStyleForGamepadType(gamepad->type);
    }
}

static void SDL_FixupHIDAPIMapping(SDL_Gamepad *gamepad)
{
    // Check to see if we need fixup
    bool need_fixup = false;
    for (int i = 0; i < gamepad->num_bindings; ++i) {
        SDL_GamepadBinding *binding = &gamepad->bindings[i];
        if (binding->output_type == SDL_GAMEPAD_BINDTYPE_BUTTON &&
            binding->output.button >= SDL_GAMEPAD_BUTTON_DPAD_UP) {
            if (binding->input_type == SDL_GAMEPAD_BINDTYPE_BUTTON &&
                binding->input.button == binding->output.button) {
                // Old style binding
                need_fixup = true;
            }
            break;
        }
    }
    if (!need_fixup) {
        return;
    }

    for (int i = 0; i < gamepad->num_bindings; ++i) {
        SDL_GamepadBinding *binding = &gamepad->bindings[i];
        if (binding->input_type == SDL_GAMEPAD_BINDTYPE_BUTTON &&
            binding->output_type == SDL_GAMEPAD_BINDTYPE_BUTTON) {
            switch (binding->output.button) {
            case SDL_GAMEPAD_BUTTON_DPAD_UP:
                binding->input_type = SDL_GAMEPAD_BINDTYPE_HAT;
                binding->input.hat.hat = 0;
                binding->input.hat.hat_mask = SDL_HAT_UP;
                break;
            case SDL_GAMEPAD_BUTTON_DPAD_DOWN:
                binding->input_type = SDL_GAMEPAD_BINDTYPE_HAT;
                binding->input.hat.hat = 0;
                binding->input.hat.hat_mask = SDL_HAT_DOWN;
                break;
            case SDL_GAMEPAD_BUTTON_DPAD_LEFT:
                binding->input_type = SDL_GAMEPAD_BINDTYPE_HAT;
                binding->input.hat.hat = 0;
                binding->input.hat.hat_mask = SDL_HAT_LEFT;
                break;
            case SDL_GAMEPAD_BUTTON_DPAD_RIGHT:
                binding->input_type = SDL_GAMEPAD_BINDTYPE_HAT;
                binding->input.hat.hat = 0;
                binding->input.hat.hat_mask = SDL_HAT_RIGHT;
                break;
            default:
                if (binding->output.button > SDL_GAMEPAD_BUTTON_DPAD_RIGHT) {
                    binding->input.button -= 4;
                }
                break;
            }
        }
    }
}

/*
 * Make a new button mapping struct
 */
static void SDL_PrivateLoadButtonMapping(SDL_Gamepad *gamepad, GamepadMapping_t *pGamepadMapping)
{
    int i;

    SDL_AssertJoysticksLocked();

    gamepad->name = pGamepadMapping->name;
    gamepad->num_bindings = 0;
    gamepad->mapping = pGamepadMapping;
    if (gamepad->joystick->naxes != 0 && gamepad->last_match_axis) {
        SDL_memset(gamepad->last_match_axis, 0, gamepad->joystick->naxes * sizeof(*gamepad->last_match_axis));
    }

    SDL_UpdateGamepadType(gamepad);
    SDL_UpdateGamepadFaceStyle(gamepad);

    SDL_PrivateParseGamepadConfigString(gamepad, pGamepadMapping->mapping);

    if (SDL_IsJoystickHIDAPI(pGamepadMapping->guid)) {
        SDL_FixupHIDAPIMapping(gamepad);
    }

    // Set the zero point for triggers
    for (i = 0; i < gamepad->num_bindings; ++i) {
        SDL_GamepadBinding *binding = &gamepad->bindings[i];
        if (binding->input_type == SDL_GAMEPAD_BINDTYPE_AXIS &&
            binding->output_type == SDL_GAMEPAD_BINDTYPE_AXIS &&
            (binding->output.axis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER ||
             binding->output.axis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)) {
            if (binding->input.axis.axis < gamepad->joystick->naxes) {
                gamepad->joystick->axes[binding->input.axis.axis].value =
                    gamepad->joystick->axes[binding->input.axis.axis].zero = (Sint16)binding->input.axis.axis_min;
            }
        }
    }
}

/*
 * grab the guid string from a mapping string
 */
static char *SDL_PrivateGetGamepadGUIDFromMappingString(const char *pMapping)
{
    const char *pFirstComma = SDL_strchr(pMapping, ',');
    if (pFirstComma) {
        char *pchGUID = (char *)SDL_malloc(pFirstComma - pMapping + 1);
        if (!pchGUID) {
            return NULL;
        }
        SDL_memcpy(pchGUID, pMapping, pFirstComma - pMapping);
        pchGUID[pFirstComma - pMapping] = '\0';

        // Convert old style GUIDs to the new style in 2.0.5
#if defined(SDL_PLATFORM_WIN32) || defined(SDL_PLATFORM_WINGDK)
        if (SDL_strlen(pchGUID) == 32 &&
            SDL_memcmp(&pchGUID[20], "504944564944", 12) == 0) {
            SDL_memcpy(&pchGUID[20], "000000000000", 12);
            SDL_memcpy(&pchGUID[16], &pchGUID[4], 4);
            SDL_memcpy(&pchGUID[8], &pchGUID[0], 4);
            SDL_memcpy(&pchGUID[0], "03000000", 8);
        }
#elif defined(SDL_PLATFORM_MACOS)
        if (SDL_strlen(pchGUID) == 32 &&
            SDL_memcmp(&pchGUID[4], "000000000000", 12) == 0 &&
            SDL_memcmp(&pchGUID[20], "000000000000", 12) == 0) {
            SDL_memcpy(&pchGUID[20], "000000000000", 12);
            SDL_memcpy(&pchGUID[8], &pchGUID[0], 4);
            SDL_memcpy(&pchGUID[0], "03000000", 8);
        }
#endif
        return pchGUID;
    }
    return NULL;
}

/*
 * grab the name string from a mapping string
 */
static char *SDL_PrivateGetGamepadNameFromMappingString(const char *pMapping)
{
    const char *pFirstComma, *pSecondComma;
    char *pchName;

    pFirstComma = SDL_strchr(pMapping, ',');
    if (!pFirstComma) {
        return NULL;
    }

    pSecondComma = SDL_strchr(pFirstComma + 1, ',');
    if (!pSecondComma) {
        return NULL;
    }

    pchName = (char *)SDL_malloc(pSecondComma - pFirstComma);
    if (!pchName) {
        return NULL;
    }
    SDL_memcpy(pchName, pFirstComma + 1, pSecondComma - pFirstComma);
    pchName[pSecondComma - pFirstComma - 1] = 0;
    return pchName;
}

/*
 * grab the button mapping string from a mapping string
 */
static char *SDL_PrivateGetGamepadMappingFromMappingString(const char *pMapping)
{
    const char *pFirstComma, *pSecondComma;
    char *result;
    size_t length;

    pFirstComma = SDL_strchr(pMapping, ',');
    if (!pFirstComma) {
        return NULL;
    }

    pSecondComma = SDL_strchr(pFirstComma + 1, ',');
    if (!pSecondComma) {
        return NULL;
    }

    // Skip whitespace
    while (SDL_isspace(pSecondComma[1])) {
        ++pSecondComma;
    }

    result = SDL_strdup(pSecondComma + 1); // mapping is everything after the 3rd comma

    // Trim whitespace
    length = SDL_strlen(result);
    while (length > 0 && SDL_isspace(result[length - 1])) {
        --length;
    }
    result[length] = '\0';

    return result;
}

/*
 * Helper function to add a mapping for a guid
 */
static GamepadMapping_t *SDL_PrivateAddMappingForGUID(SDL_GUID jGUID, const char *mappingString, bool *existing, SDL_GamepadMappingPriority priority)
{
    char *pchName;
    char *pchMapping;
    GamepadMapping_t *pGamepadMapping;
    Uint16 crc;

    SDL_AssertJoysticksLocked();

    pchName = SDL_PrivateGetGamepadNameFromMappingString(mappingString);
    if (!pchName) {
        SDL_SetError("Couldn't parse name from %s", mappingString);
        return NULL;
    }

    pchMapping = SDL_PrivateGetGamepadMappingFromMappingString(mappingString);
    if (!pchMapping) {
        SDL_free(pchName);
        SDL_SetError("Couldn't parse %s", mappingString);
        return NULL;
    }

    // Fix up the GUID and the mapping with the CRC, if needed
    SDL_GetJoystickGUIDInfo(jGUID, NULL, NULL, NULL, &crc);
    if (crc) {
        // Make sure the mapping has the CRC
        char *new_mapping;
        const char *optional_comma;
        size_t mapping_length;
        char *crc_end = "";
        char *crc_string = SDL_strstr(pchMapping, SDL_GAMEPAD_CRC_FIELD);
        if (crc_string) {
            crc_end = SDL_strchr(crc_string, ',');
            if (crc_end) {
                ++crc_end;
            } else {
                crc_end = "";
            }
            *crc_string = '\0';
        }

        // Make sure there's a comma before the CRC
        mapping_length = SDL_strlen(pchMapping);
        if (mapping_length == 0 || pchMapping[mapping_length - 1] == ',') {
            optional_comma = "";
        } else {
            optional_comma = ",";
        }

        if (SDL_asprintf(&new_mapping, "%s%s%s%.4x,%s", pchMapping, optional_comma, SDL_GAMEPAD_CRC_FIELD, crc, crc_end) >= 0) {
            SDL_free(pchMapping);
            pchMapping = new_mapping;
        }
    } else {
        // Make sure the GUID has the CRC, for matching purposes
        char *crc_string = SDL_strstr(pchMapping, SDL_GAMEPAD_CRC_FIELD);
        if (crc_string) {
            crc = (Uint16)SDL_strtol(crc_string + SDL_GAMEPAD_CRC_FIELD_SIZE, NULL, 16);
            if (crc) {
                SDL_SetJoystickGUIDCRC(&jGUID, crc);
            }
        }
    }

    PushMappingChangeTracking();

    pGamepadMapping = SDL_PrivateGetGamepadMappingForGUID(jGUID, true);
    if (pGamepadMapping) {
        // Only overwrite the mapping if the priority is the same or higher.
        if (pGamepadMapping->priority <= priority) {
            // Update existing mapping
            SDL_free(pGamepadMapping->name);
            pGamepadMapping->name = pchName;
            SDL_free(pGamepadMapping->mapping);
            pGamepadMapping->mapping = pchMapping;
            pGamepadMapping->priority = priority;
        } else {
            SDL_free(pchName);
            SDL_free(pchMapping);
        }
        if (existing) {
            *existing = true;
        }
        AddMappingChangeTracking(pGamepadMapping);
    } else {
        pGamepadMapping = (GamepadMapping_t *)SDL_malloc(sizeof(*pGamepadMapping));
        if (!pGamepadMapping) {
            PopMappingChangeTracking();
            SDL_free(pchName);
            SDL_free(pchMapping);
            return NULL;
        }
        // Clear the CRC, we've already added it to the mapping
        if (crc) {
            SDL_SetJoystickGUIDCRC(&jGUID, 0);
        }
        pGamepadMapping->guid = jGUID;
        pGamepadMapping->name = pchName;
        pGamepadMapping->mapping = pchMapping;
        pGamepadMapping->next = NULL;
        pGamepadMapping->priority = priority;

        if (s_pSupportedGamepads) {
            // Add the mapping to the end of the list
            GamepadMapping_t *pCurrMapping, *pPrevMapping;

            for (pPrevMapping = s_pSupportedGamepads, pCurrMapping = pPrevMapping->next;
                 pCurrMapping;
                 pPrevMapping = pCurrMapping, pCurrMapping = pCurrMapping->next) {
                // continue;
            }
            pPrevMapping->next = pGamepadMapping;
        } else {
            s_pSupportedGamepads = pGamepadMapping;
        }
        if (existing) {
            *existing = false;
        }
    }

    PopMappingChangeTracking();

    return pGamepadMapping;
}

/*
 * Helper function to determine pre-calculated offset to certain joystick mappings
 */
static GamepadMapping_t *SDL_PrivateGetGamepadMappingForNameAndGUID(const char *name, SDL_GUID guid)
{
    GamepadMapping_t *mapping;

    SDL_AssertJoysticksLocked();

    mapping = SDL_PrivateGetGamepadMappingForGUID(guid, false);

    return mapping;
}

static void SDL_PrivateAppendToMappingString(char *mapping_string,
                                             size_t mapping_string_len,
                                             const char *input_name,
                                             SDL_InputMapping *mapping)
{
    char buffer[16];
    if (mapping->kind == EMappingKind_None) {
        return;
    }

    SDL_strlcat(mapping_string, input_name, mapping_string_len);
    SDL_strlcat(mapping_string, ":", mapping_string_len);
    switch (mapping->kind) {
    case EMappingKind_Button:
        (void)SDL_snprintf(buffer, sizeof(buffer), "b%u", mapping->target);
        break;
    case EMappingKind_Axis:
        (void)SDL_snprintf(buffer, sizeof(buffer), "%sa%u%s",
            mapping->half_axis_positive ? "+" :
            mapping->half_axis_negative ? "-" : "",
            mapping->target,
            mapping->axis_reversed ? "~" : "");
        break;
    case EMappingKind_Hat:
        (void)SDL_snprintf(buffer, sizeof(buffer), "h%i.%i", mapping->target >> 4, mapping->target & 0x0F);
        break;
    default:
        SDL_assert(false);
    }

    SDL_strlcat(mapping_string, buffer, mapping_string_len);
    SDL_strlcat(mapping_string, ",", mapping_string_len);
}

static GamepadMapping_t *SDL_PrivateGenerateAutomaticGamepadMapping(const char *name,
                                                                          SDL_GUID guid,
                                                                          SDL_GamepadMapping *raw_map)
{
    bool existing;
    char name_string[128];
    char mapping[1024];

    // Remove the CRC from the GUID
    // We already know that this GUID doesn't have a mapping without the CRC, and we want newly
    // added mappings without a CRC to override this mapping.
    SDL_SetJoystickGUIDCRC(&guid, 0);

    // Remove any commas in the name
    SDL_strlcpy(name_string, name, sizeof(name_string));
    {
        char *spot;
        for (spot = name_string; *spot; ++spot) {
            if (*spot == ',') {
                *spot = ' ';
            }
        }
    }
    (void)SDL_snprintf(mapping, sizeof(mapping), "none,%s,", name_string);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "a", &raw_map->a);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "b", &raw_map->b);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "x", &raw_map->x);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "y", &raw_map->y);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "back", &raw_map->back);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "guide", &raw_map->guide);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "start", &raw_map->start);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "leftstick", &raw_map->leftstick);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "rightstick", &raw_map->rightstick);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "leftshoulder", &raw_map->leftshoulder);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "rightshoulder", &raw_map->rightshoulder);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "dpup", &raw_map->dpup);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "dpdown", &raw_map->dpdown);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "dpleft", &raw_map->dpleft);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "dpright", &raw_map->dpright);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "misc1", &raw_map->misc1);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "misc2", &raw_map->misc2);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "misc3", &raw_map->misc3);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "misc4", &raw_map->misc4);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "misc5", &raw_map->misc5);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "misc6", &raw_map->misc6);
    /* Keep using paddle1-4 in the generated mapping so that it can be
     * reused with SDL2 */
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "paddle1", &raw_map->right_paddle1);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "paddle2", &raw_map->left_paddle1);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "paddle3", &raw_map->right_paddle2);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "paddle4", &raw_map->left_paddle2);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "leftx", &raw_map->leftx);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "lefty", &raw_map->lefty);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "rightx", &raw_map->rightx);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "righty", &raw_map->righty);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "lefttrigger", &raw_map->lefttrigger);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "righttrigger", &raw_map->righttrigger);
    SDL_PrivateAppendToMappingString(mapping, sizeof(mapping), "touchpad", &raw_map->touchpad);

    return SDL_PrivateAddMappingForGUID(guid, mapping, &existing, SDL_GAMEPAD_MAPPING_PRIORITY_DEFAULT);
}

static GamepadMapping_t *SDL_PrivateGetGamepadMapping(SDL_JoystickID instance_id, bool create_mapping)
{
    const char *name;
    SDL_GUID guid;
    GamepadMapping_t *mapping;

    SDL_AssertJoysticksLocked();

    name = SDL_GetJoystickNameForID(instance_id);
    guid = SDL_GetJoystickGUIDForID(instance_id);
    mapping = SDL_PrivateGetGamepadMappingForNameAndGUID(name, guid);
    if (!mapping && create_mapping) {
        SDL_GamepadMapping raw_map;

        SDL_zero(raw_map);
        if (SDL_PrivateJoystickGetAutoGamepadMapping(instance_id, &raw_map)) {
            mapping = SDL_PrivateGenerateAutomaticGamepadMapping(name, guid, &raw_map);
        }
    }

    if (!mapping) {
        mapping = s_pDefaultMapping;
    }
    return mapping;
}

/*
 * Add or update an entry into the Mappings Database
 */
int SDL_AddGamepadMappingsFromIO(SDL_IOStream *src, bool closeio)
{
    const char *platform = SDL_GetPlatform();
    int gamepads = 0;
    char *buf, *line, *line_end, *tmp, *comma, line_platform[64];
    size_t db_size;
    size_t platform_len;

    buf = (char *)SDL_LoadFile_IO(src, &db_size, closeio);
    if (!buf) {
        SDL_SetError("Could not allocate space to read DB into memory");
        return -1;
    }
    line = buf;

    SDL_LockJoysticks();

    PushMappingChangeTracking();

    while (line < buf + db_size) {
        line_end = SDL_strchr(line, '\n');
        if (line_end) {
            *line_end = '\0';
        } else {
            line_end = buf + db_size;
        }

        // Extract and verify the platform
        tmp = SDL_strstr(line, SDL_GAMEPAD_PLATFORM_FIELD);
        if (tmp) {
            tmp += SDL_GAMEPAD_PLATFORM_FIELD_SIZE;
            comma = SDL_strchr(tmp, ',');
            if (comma) {
                platform_len = comma - tmp + 1;
                if (platform_len + 1 < SDL_arraysize(line_platform)) {
                    SDL_strlcpy(line_platform, tmp, platform_len);
                    if (SDL_strncasecmp(line_platform, platform, platform_len) == 0 &&
                        SDL_AddGamepadMapping(line) > 0) {
                        gamepads++;
                    }
                }
            }
        }

        line = line_end + 1;
    }

    PopMappingChangeTracking();

    SDL_UnlockJoysticks();

    SDL_free(buf);
    return gamepads;
}

int SDL_AddGamepadMappingsFromFile(const char *file)
{
    return SDL_AddGamepadMappingsFromIO(SDL_IOFromFile(file, "rb"), true);
}

bool SDL_ReloadGamepadMappings(void)
{
    SDL_Gamepad *gamepad;

    SDL_LockJoysticks();

    PushMappingChangeTracking();

    for (gamepad = SDL_gamepads; gamepad; gamepad = gamepad->next) {
        AddMappingChangeTracking(gamepad->mapping);
    }

    SDL_QuitGamepadMappings();
    SDL_InitGamepadMappings();

    PopMappingChangeTracking();

    SDL_UnlockJoysticks();

    return true;
}

static char *SDL_ConvertMappingToPositional(const char *mapping)
{
    // Add space for '!' and null terminator
    size_t length = SDL_strlen(mapping) + 1 + 1;
    char *remapped = (char *)SDL_malloc(length);
    if (remapped) {
        char *button_A;
        char *button_B;
        char *button_X;
        char *button_Y;
        char *hint;

        SDL_strlcpy(remapped, mapping, length);
        button_A = SDL_strstr(remapped, "a:");
        button_B = SDL_strstr(remapped, "b:");
        button_X = SDL_strstr(remapped, "x:");
        button_Y = SDL_strstr(remapped, "y:");
        hint = SDL_strstr(remapped, "hint:SDL_GAMECONTROLLER_USE_BUTTON_LABELS");

        if (button_A) {
            *button_A = 'b';
        }
        if (button_B) {
            *button_B = 'a';
        }
        if (button_X) {
            *button_X = 'y';
        }
        if (button_Y) {
            *button_Y = 'x';
        }
        if (hint) {
            hint += 5;
            SDL_memmove(hint + 1, hint, SDL_strlen(hint) + 1);
            *hint = '!';
        }
    }
    return remapped;
}

/*
 * Add or update an entry into the Mappings Database with a priority
 */
static int SDL_PrivateAddGamepadMapping(const char *mappingString, SDL_GamepadMappingPriority priority)
{
    char *remapped = NULL;
    char *pchGUID;
    SDL_GUID jGUID;
    bool is_default_mapping = false;
    bool is_xinput_mapping = false;
    bool existing = false;
    GamepadMapping_t *pGamepadMapping;
    int result = -1;

    SDL_AssertJoysticksLocked();

    if (!mappingString) {
        SDL_InvalidParamError("mappingString");
        return -1;
    }

    { // Extract and verify the hint field
        const char *tmp;

        tmp = SDL_strstr(mappingString, SDL_GAMEPAD_HINT_FIELD);
        if (tmp) {
            bool default_value, value, negate;
            int len;
            char hint[128];

            tmp += SDL_GAMEPAD_HINT_FIELD_SIZE;

            if (*tmp == '!') {
                negate = true;
                ++tmp;
            } else {
                negate = false;
            }

            len = 0;
            while (*tmp && *tmp != ',' && *tmp != ':' && len < (sizeof(hint) - 1)) {
                hint[len++] = *tmp++;
            }
            hint[len] = '\0';

            if (tmp[0] == ':' && tmp[1] == '=') {
                tmp += 2;
                default_value = SDL_atoi(tmp);
            } else {
                default_value = false;
            }

            if (SDL_strcmp(hint, "SDL_GAMECONTROLLER_USE_BUTTON_LABELS") == 0) {
                // This hint is used to signal whether the mapping uses positional buttons or not
                if (negate) {
                    // This mapping uses positional buttons, we can use it as-is
                } else {
                    // This mapping uses labeled buttons, we need to swap them to positional
                    remapped = SDL_ConvertMappingToPositional(mappingString);
                    if (!remapped) {
                        goto done;
                    }
                    mappingString = remapped;
                }
            } else {
                value = SDL_GetHintBoolean(hint, default_value);
                if (negate) {
                    value = !value;
                }
                if (!value) {
                    result = 0;
                    goto done;
                }
            }
        }
    }

#ifdef ANDROID
    { // Extract and verify the SDK version
        const char *tmp;

        tmp = SDL_strstr(mappingString, SDL_GAMEPAD_SDKGE_FIELD);
        if (tmp) {
            tmp += SDL_GAMEPAD_SDKGE_FIELD_SIZE;
            if (!(SDL_GetAndroidSDKVersion() >= SDL_atoi(tmp))) {
                SDL_SetError("SDK version %d < minimum version %d", SDL_GetAndroidSDKVersion(), SDL_atoi(tmp));
                goto done;
            }
        }
        tmp = SDL_strstr(mappingString, SDL_GAMEPAD_SDKLE_FIELD);
        if (tmp) {
            tmp += SDL_GAMEPAD_SDKLE_FIELD_SIZE;
            if (!(SDL_GetAndroidSDKVersion() <= SDL_atoi(tmp))) {
                SDL_SetError("SDK version %d > maximum version %d", SDL_GetAndroidSDKVersion(), SDL_atoi(tmp));
                goto done;
            }
        }
    }
#endif

    pchGUID = SDL_PrivateGetGamepadGUIDFromMappingString(mappingString);
    if (!pchGUID) {
        SDL_SetError("Couldn't parse GUID from %s", mappingString);
        goto done;
    }
    if (!SDL_strcasecmp(pchGUID, "default")) {
        is_default_mapping = true;
    } else if (!SDL_strcasecmp(pchGUID, "xinput")) {
        is_xinput_mapping = true;
    }
    jGUID = SDL_StringToGUID(pchGUID);
    SDL_free(pchGUID);

    pGamepadMapping = SDL_PrivateAddMappingForGUID(jGUID, mappingString, &existing, priority);
    if (!pGamepadMapping) {
        goto done;
    }

    if (existing) {
        result = 0;
    } else {
        if (is_default_mapping) {
            s_pDefaultMapping = pGamepadMapping;
        } else if (is_xinput_mapping) {
            s_pXInputMapping = pGamepadMapping;
        }
        result = 1;
    }
done:
    if (remapped) {
        SDL_free(remapped);
    }
    return result;
}

/*
 * Add or update an entry into the Mappings Database
 */
int SDL_AddGamepadMapping(const char *mapping)
{
    int result;

    SDL_LockJoysticks();
    {
        result = SDL_PrivateAddGamepadMapping(mapping, SDL_GAMEPAD_MAPPING_PRIORITY_API);
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 * Create a mapping string for a mapping
 */
static char *CreateMappingString(GamepadMapping_t *mapping, SDL_GUID guid)
{
    char *pMappingString, *pPlatformString;
    char pchGUID[33];
    size_t needed;
    bool need_platform = false;
    const char *platform = NULL;

    SDL_AssertJoysticksLocked();

    SDL_GUIDToString(guid, pchGUID, sizeof(pchGUID));

    // allocate enough memory for GUID + ',' + name + ',' + mapping + \0
    needed = SDL_strlen(pchGUID) + 1 + SDL_strlen(mapping->name) + 1 + SDL_strlen(mapping->mapping) + 1;

    if (!SDL_strstr(mapping->mapping, SDL_GAMEPAD_PLATFORM_FIELD)) {
        // add memory for ',' + platform:PLATFORM
        need_platform = true;
        if (mapping->mapping[SDL_strlen(mapping->mapping) - 1] != ',') {
            needed += 1;
        }
        platform = SDL_GetPlatform();
        needed += SDL_GAMEPAD_PLATFORM_FIELD_SIZE + SDL_strlen(platform) + 1;
    }

    pMappingString = (char *)SDL_malloc(needed);
    if (!pMappingString) {
        return NULL;
    }

    (void)SDL_snprintf(pMappingString, needed, "%s,%s,%s", pchGUID, mapping->name, mapping->mapping);

    if (need_platform) {
        if (mapping->mapping[SDL_strlen(mapping->mapping) - 1] != ',') {
            SDL_strlcat(pMappingString, ",", needed);
        }
        SDL_strlcat(pMappingString, SDL_GAMEPAD_PLATFORM_FIELD, needed);
        SDL_strlcat(pMappingString, platform, needed);
        SDL_strlcat(pMappingString, ",", needed);
    }

    // Make sure multiple platform strings haven't made their way into the mapping
    pPlatformString = SDL_strstr(pMappingString, SDL_GAMEPAD_PLATFORM_FIELD);
    if (pPlatformString) {
        pPlatformString = SDL_strstr(pPlatformString + 1, SDL_GAMEPAD_PLATFORM_FIELD);
        if (pPlatformString) {
            *pPlatformString = '\0';
        }
    }
    return pMappingString;
}

char **SDL_GetGamepadMappings(int *count)
{
    int num_mappings = 0;
    char **result = NULL;
    char **mappings = NULL;

    if (count) {
        *count = 0;
    }

    SDL_LockJoysticks();

    for (GamepadMapping_t *mapping = s_pSupportedGamepads; mapping; mapping = mapping->next) {
        if (SDL_memcmp(&mapping->guid, &s_zeroGUID, sizeof(mapping->guid)) == 0) {
            continue;
        }
        num_mappings++;
    }

    size_t final_allocation = sizeof (char *);  // for the NULL terminator element.
    bool failed = false;
    mappings = (char **) SDL_calloc(num_mappings + 1, sizeof (char *));
    if (!mappings) {
        failed = true;
    } else {
        int i = 0;
        for (GamepadMapping_t *mapping = s_pSupportedGamepads; mapping; mapping = mapping->next) {
            if (SDL_memcmp(&mapping->guid, &s_zeroGUID, sizeof(mapping->guid)) == 0) {
                continue;
            }

            char *mappingstr = CreateMappingString(mapping, mapping->guid);
            if (!mappingstr) {
                failed = true;
                break;  // error string is already set.
            }

            SDL_assert(i < num_mappings);
            mappings[i++] = mappingstr;

            final_allocation += SDL_strlen(mappingstr) + 1 + sizeof (char *);
        }
    }

    SDL_UnlockJoysticks();

    if (!failed) {
        result = (char **) SDL_malloc(final_allocation);
        if (result) {
            final_allocation -= (sizeof (char *) * num_mappings + 1);
            char *strptr = (char *) (result + (num_mappings + 1));
            for (int i = 0; i < num_mappings; i++) {
                result[i] = strptr;
                const size_t slen = SDL_strlcpy(strptr, mappings[i], final_allocation) + 1;
                SDL_assert(final_allocation >= slen);
                final_allocation -= slen;
                strptr += slen;
            }
            result[num_mappings] = NULL;

            if (count) {
                *count = num_mappings;
            }
        }
    }

    if (mappings) {
        for (int i = 0; i < num_mappings; i++) {
            SDL_free(mappings[i]);
        }
        SDL_free(mappings);
    }

    return result;
}

/*
 * Get the mapping string for this GUID
 */
char *SDL_GetGamepadMappingForGUID(SDL_GUID guid)
{
    char *result;

    SDL_LockJoysticks();
    {
        GamepadMapping_t *mapping = SDL_PrivateGetGamepadMappingForGUID(guid, false);
        if (mapping) {
            result = CreateMappingString(mapping, guid);
        } else {
            SDL_SetError("Mapping not available");
            result = NULL;
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 * Get the mapping string for this device
 */
char *SDL_GetGamepadMapping(SDL_Gamepad *gamepad)
{
    char *result;

    SDL_LockJoysticks();
    {
        CHECK_GAMEPAD_MAGIC(gamepad, NULL);

        result = CreateMappingString(gamepad->mapping, gamepad->joystick->guid);
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 * Set the mapping string for this device
 */
bool SDL_SetGamepadMapping(SDL_JoystickID instance_id, const char *mapping)
{
    SDL_GUID guid = SDL_GetJoystickGUIDForID(instance_id);
    bool result = false;

    if (SDL_memcmp(&guid, &s_zeroGUID, sizeof(guid)) == 0) {
        return SDL_InvalidParamError("instance_id");
    }

    if (!mapping) {
        mapping = "*,*,";
    }

    SDL_LockJoysticks();
    {
        if (SDL_PrivateAddMappingForGUID(guid, mapping, NULL, SDL_GAMEPAD_MAPPING_PRIORITY_API)) {
            result = true;
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

static void SDL_LoadGamepadHints(void)
{
    const char *hint = SDL_GetHint(SDL_HINT_GAMECONTROLLERCONFIG);
    if (hint && hint[0]) {
        char *pTempMappings = SDL_strdup(hint);
        char *pUserMappings = pTempMappings;

        PushMappingChangeTracking();

        while (pUserMappings) {
            char *pchNewLine = NULL;

            pchNewLine = SDL_strchr(pUserMappings, '\n');
            if (pchNewLine) {
                *pchNewLine = '\0';
            }

            SDL_PrivateAddGamepadMapping(pUserMappings, SDL_GAMEPAD_MAPPING_PRIORITY_USER);

            if (pchNewLine) {
                pUserMappings = pchNewLine + 1;
            } else {
                pUserMappings = NULL;
            }
        }

        PopMappingChangeTracking();

        SDL_free(pTempMappings);
    }
}

/*
 * Fill the given buffer with the expected gamepad mapping filepath.
 * Usually this will just be SDL_HINT_GAMECONTROLLERCONFIG_FILE, but for
 * Android, we want to get the internal storage path.
 */
static bool SDL_GetGamepadMappingFilePath(char *path, size_t size)
{
    const char *hint = SDL_GetHint(SDL_HINT_GAMECONTROLLERCONFIG_FILE);
    if (hint && *hint) {
        return SDL_strlcpy(path, hint, size) < size;
    }

#ifdef SDL_PLATFORM_ANDROID
    return SDL_snprintf(path, size, "%s/gamepad_map.txt", SDL_GetAndroidInternalStoragePath()) < size;
#else
    return false;
#endif
}

/*
 * Initialize the gamepad system, mostly load our DB of gamepad config mappings
 */
bool SDL_InitGamepadMappings(void)
{
    char szGamepadMapPath[1024];
    int i = 0;
    const char *pMappingString = NULL;

    SDL_AssertJoysticksLocked();

    PushMappingChangeTracking();

    pMappingString = s_GamepadMappings[i];
    while (pMappingString) {
        SDL_PrivateAddGamepadMapping(pMappingString, SDL_GAMEPAD_MAPPING_PRIORITY_DEFAULT);

        i++;
        pMappingString = s_GamepadMappings[i];
    }

    if (SDL_GetGamepadMappingFilePath(szGamepadMapPath, sizeof(szGamepadMapPath))) {
        SDL_AddGamepadMappingsFromFile(szGamepadMapPath);
    }

    // load in any user supplied config
    SDL_LoadGamepadHints();

    SDL_LoadVIDPIDList(&SDL_allowed_gamepads);
    SDL_LoadVIDPIDList(&SDL_ignored_gamepads);

    PopMappingChangeTracking();

    return true;
}

bool SDL_InitGamepads(void)
{
    int i;
    SDL_JoystickID *joysticks;

    SDL_gamepads_initialized = true;

    // Watch for joystick events and fire gamepad ones if needed
    SDL_AddEventWatch(SDL_GamepadEventWatcher, NULL);

    // Send added events for gamepads currently attached
    joysticks = SDL_GetJoysticks(NULL);
    if (joysticks) {
        for (i = 0; joysticks[i]; ++i) {
            if (SDL_IsGamepad(joysticks[i])) {
                SDL_PrivateGamepadAdded(joysticks[i]);
            }
        }
        SDL_free(joysticks);
    }

    return true;
}

bool SDL_HasGamepad(void)
{
    int num_joysticks = 0;
    int num_gamepads = 0;
    SDL_JoystickID *joysticks = SDL_GetJoysticks(&num_joysticks);
    if (joysticks) {
        int i;
        for (i = num_joysticks - 1; i >= 0 && num_gamepads == 0; --i) {
            if (SDL_IsGamepad(joysticks[i])) {
                ++num_gamepads;
            }
        }
        SDL_free(joysticks);
    }
    if (num_gamepads > 0) {
        return true;
    }
    return false;
}

SDL_JoystickID *SDL_GetGamepads(int *count)
{
    int num_joysticks = 0;
    int num_gamepads = 0;
    SDL_JoystickID *joysticks = SDL_GetJoysticks(&num_joysticks);
    if (joysticks) {
        int i;
        for (i = num_joysticks - 1; i >= 0; --i) {
            if (SDL_IsGamepad(joysticks[i])) {
                ++num_gamepads;
            } else {
                SDL_memmove(&joysticks[i], &joysticks[i+1], (num_gamepads + 1) * sizeof(joysticks[i]));
            }
        }
    }
    if (count) {
        *count = num_gamepads;
    }
    return joysticks;
}

const char *SDL_GetGamepadNameForID(SDL_JoystickID instance_id)
{
    const char *result = NULL;

    SDL_LockJoysticks();
    {
        GamepadMapping_t *mapping = SDL_PrivateGetGamepadMapping(instance_id, true);
        if (mapping) {
            if (SDL_strcmp(mapping->name, "*") == 0) {
                result = SDL_GetJoystickNameForID(instance_id);
            } else {
                result = SDL_GetPersistentString(mapping->name);
            }
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

const char *SDL_GetGamepadPathForID(SDL_JoystickID instance_id)
{
    return SDL_GetJoystickPathForID(instance_id);
}

int SDL_GetGamepadPlayerIndexForID(SDL_JoystickID instance_id)
{
    return SDL_GetJoystickPlayerIndexForID(instance_id);
}

SDL_GUID SDL_GetGamepadGUIDForID(SDL_JoystickID instance_id)
{
    return SDL_GetJoystickGUIDForID(instance_id);
}

Uint16 SDL_GetGamepadVendorForID(SDL_JoystickID instance_id)
{
    return SDL_GetJoystickVendorForID(instance_id);
}

Uint16 SDL_GetGamepadProductForID(SDL_JoystickID instance_id)
{
    return SDL_GetJoystickProductForID(instance_id);
}

Uint16 SDL_GetGamepadProductVersionForID(SDL_JoystickID instance_id)
{
    return SDL_GetJoystickProductVersionForID(instance_id);
}

SDL_GamepadType SDL_GetGamepadTypeForID(SDL_JoystickID instance_id)
{
    SDL_GamepadType type = SDL_GAMEPAD_TYPE_UNKNOWN;

    SDL_LockJoysticks();
    {
        GamepadMapping_t *mapping = SDL_PrivateGetGamepadMapping(instance_id, true);
        if (mapping) {
            char *type_string, *comma;

            type_string = SDL_strstr(mapping->mapping, SDL_GAMEPAD_TYPE_FIELD);
            if (type_string) {
                type_string += SDL_GAMEPAD_TYPE_FIELD_SIZE;
                comma = SDL_strchr(type_string, ',');
                if (comma) {
                    *comma = '\0';
                    type = SDL_GetGamepadTypeFromString(type_string);
                    *comma = ',';
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    if (type != SDL_GAMEPAD_TYPE_UNKNOWN) {
        return type;
    }
    return SDL_GetRealGamepadTypeForID(instance_id);
}

SDL_GamepadType SDL_GetRealGamepadTypeForID(SDL_JoystickID instance_id)
{
    SDL_GamepadType type = SDL_GAMEPAD_TYPE_UNKNOWN;
    const SDL_SteamVirtualGamepadInfo *info;

    SDL_LockJoysticks();
    {
        info = SDL_GetJoystickVirtualGamepadInfoForID(instance_id);
        if (info) {
            type = info->type;
        } else {
            type = SDL_GetGamepadTypeFromGUID(SDL_GetJoystickGUIDForID(instance_id), SDL_GetJoystickNameForID(instance_id));
        }
    }
    SDL_UnlockJoysticks();

    return type;
}

char *SDL_GetGamepadMappingForID(SDL_JoystickID instance_id)
{
    char *result = NULL;

    SDL_LockJoysticks();
    {
        GamepadMapping_t *mapping = SDL_PrivateGetGamepadMapping(instance_id, true);
        if (mapping) {
            char pchGUID[33];
            SDL_GUID guid = SDL_GetJoystickGUIDForID(instance_id);
            SDL_GUIDToString(guid, pchGUID, sizeof(pchGUID));
            SDL_asprintf(&result, "%s,%s,%s", pchGUID, mapping->name, mapping->mapping);
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 * Return 1 if the joystick with this name and GUID is a supported gamepad
 */
bool SDL_IsGamepadNameAndGUID(const char *name, SDL_GUID guid)
{
    bool result;

    SDL_LockJoysticks();
    {
        if (s_pDefaultMapping || SDL_PrivateGetGamepadMappingForNameAndGUID(name, guid) != NULL) {
            result = true;
        } else {
            result = false;
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 * Return 1 if the joystick at this device index is a supported gamepad
 */
bool SDL_IsGamepad(SDL_JoystickID instance_id)
{
    bool result;

    SDL_LockJoysticks();
    {
        const void *value;
        if (SDL_FindInHashTable(s_gamepadInstanceIDs, (void *)(uintptr_t)instance_id, &value)) {
            result = (bool)(uintptr_t)value;
        } else {
            if (SDL_PrivateGetGamepadMapping(instance_id, true) != NULL) {
                result = true;
            } else {
                result = false;
            }

            if (!s_gamepadInstanceIDs) {
                s_gamepadInstanceIDs = SDL_CreateHashTable(0, false, SDL_HashID, SDL_KeyMatchID, NULL, NULL);
            }
            SDL_InsertIntoHashTable(s_gamepadInstanceIDs, (void *)(uintptr_t)instance_id, (void *)(uintptr_t)result, true);
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 * Return 1 if the gamepad should be ignored by SDL
 */
bool SDL_ShouldIgnoreGamepad(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    int i;
    for (i = 0; i < SDL_arraysize(SDL_gamepad_blacklist_words); i++) {
        const struct SDL_GamepadBlacklistWords *blacklist_word = &SDL_gamepad_blacklist_words[i];

        switch (blacklist_word->pos) {
            case GAMEPAD_BLACKLIST_BEGIN:
                if (SDL_startswith(name, blacklist_word->str)) {
                    return true;
                }
                break;
            
            case GAMEPAD_BLACKLIST_END:
                if (SDL_endswith(name, blacklist_word->str)) {
                    return true;
                }
                break;

            case GAMEPAD_BLACKLIST_ANYWHERE:
                if (SDL_strstr(name, blacklist_word->str) != NULL) {
                    return true;
                }
                break;
        }
    }

#ifdef SDL_PLATFORM_WIN32
    if (SDL_GetHintBoolean("SDL_GAMECONTROLLER_ALLOW_STEAM_VIRTUAL_GAMEPAD", false) &&
        WIN_IsWine()) {
        // We are launched by Steam and running under Proton or Wine
        // We can't tell whether this controller is a Steam Virtual Gamepad,
        // so assume that is doing the appropriate filtering of controllers
        // and anything we see here is fine to use.
        return false;
    }
#endif // SDL_PLATFORM_WIN32

    if (SDL_IsJoystickSteamVirtualGamepad(vendor_id, product_id, version)) {
        return !SDL_GetHintBoolean("SDL_GAMECONTROLLER_ALLOW_STEAM_VIRTUAL_GAMEPAD", false);
    }

    if (SDL_allowed_gamepads.num_included_entries > 0) {
        if (SDL_VIDPIDInList(vendor_id, product_id, &SDL_allowed_gamepads)) {
            return false;
        }
        return true;
    } else {
        if (SDL_VIDPIDInList(vendor_id, product_id, &SDL_ignored_gamepads)) {
            return true;
        }
        return false;
    }
}

/*
 * Open a gamepad for use
 *
 * This function returns a gamepad identifier, or NULL if an error occurred.
 */
SDL_Gamepad *SDL_OpenGamepad(SDL_JoystickID instance_id)
{
    SDL_Gamepad *gamepad;
    SDL_Gamepad *gamepadlist;
    GamepadMapping_t *pSupportedGamepad = NULL;

    SDL_LockJoysticks();

    gamepadlist = SDL_gamepads;
    // If the gamepad is already open, return it
    while (gamepadlist) {
        if (instance_id == gamepadlist->joystick->instance_id) {
            gamepad = gamepadlist;
            ++gamepad->ref_count;
            SDL_UnlockJoysticks();
            return gamepad;
        }
        gamepadlist = gamepadlist->next;
    }

    // Find a gamepad mapping
    pSupportedGamepad = SDL_PrivateGetGamepadMapping(instance_id, true);
    if (!pSupportedGamepad) {
        SDL_SetError("Couldn't find mapping for device (%" SDL_PRIu32 ")", instance_id);
        SDL_UnlockJoysticks();
        return NULL;
    }

    // Create and initialize the gamepad
    gamepad = (SDL_Gamepad *)SDL_calloc(1, sizeof(*gamepad));
    if (!gamepad) {
        SDL_UnlockJoysticks();
        return NULL;
    }
    SDL_SetObjectValid(gamepad, SDL_OBJECT_TYPE_GAMEPAD, true);

    gamepad->joystick = SDL_OpenJoystick(instance_id);
    if (!gamepad->joystick) {
        SDL_SetObjectValid(gamepad, SDL_OBJECT_TYPE_GAMEPAD, false);
        SDL_free(gamepad);
        SDL_UnlockJoysticks();
        return NULL;
    }

    if (gamepad->joystick->naxes) {
        gamepad->last_match_axis = (SDL_GamepadBinding **)SDL_calloc(gamepad->joystick->naxes, sizeof(*gamepad->last_match_axis));
        if (!gamepad->last_match_axis) {
            SDL_SetObjectValid(gamepad, SDL_OBJECT_TYPE_GAMEPAD, false);
            SDL_CloseJoystick(gamepad->joystick);
            SDL_free(gamepad);
            SDL_UnlockJoysticks();
            return NULL;
        }
    }
    if (gamepad->joystick->nhats) {
        gamepad->last_hat_mask = (Uint8 *)SDL_calloc(gamepad->joystick->nhats, sizeof(*gamepad->last_hat_mask));
        if (!gamepad->last_hat_mask) {
            SDL_SetObjectValid(gamepad, SDL_OBJECT_TYPE_GAMEPAD, false);
            SDL_CloseJoystick(gamepad->joystick);
            SDL_free(gamepad->last_match_axis);
            SDL_free(gamepad);
            SDL_UnlockJoysticks();
            return NULL;
        }
    }

    SDL_PrivateLoadButtonMapping(gamepad, pSupportedGamepad);

    // Add the gamepad to list
    ++gamepad->ref_count;
    // Link the gamepad in the list
    gamepad->next = SDL_gamepads;
    SDL_gamepads = gamepad;

    SDL_UnlockJoysticks();

    return gamepad;
}

/*
 * Manually pump for gamepad updates.
 */
void SDL_UpdateGamepads(void)
{
    // Just for API completeness; the joystick API does all the work.
    SDL_UpdateJoysticks();
}

/**
 *  Return whether a gamepad has a given axis
 */
bool SDL_GamepadHasAxis(SDL_Gamepad *gamepad, SDL_GamepadAxis axis)
{
    bool result = false;

    SDL_LockJoysticks();
    {
        int i;

        CHECK_GAMEPAD_MAGIC(gamepad, false);

        for (i = 0; i < gamepad->num_bindings; ++i) {
            const SDL_GamepadBinding *binding = &gamepad->bindings[i];
            if (binding->output_type == SDL_GAMEPAD_BINDTYPE_AXIS && binding->output.axis.axis == axis) {
                result = true;
                break;
            }
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 * Get the current state of an axis control on a gamepad
 */
Sint16 SDL_GetGamepadAxis(SDL_Gamepad *gamepad, SDL_GamepadAxis axis)
{
    Sint16 result = 0;

    SDL_LockJoysticks();
    {
        int i;

        CHECK_GAMEPAD_MAGIC(gamepad, 0);

        for (i = 0; i < gamepad->num_bindings; ++i) {
            const SDL_GamepadBinding *binding = &gamepad->bindings[i];
            if (binding->output_type == SDL_GAMEPAD_BINDTYPE_AXIS && binding->output.axis.axis == axis) {
                int value = 0;
                bool valid_input_range;
                bool valid_output_range;

                if (binding->input_type == SDL_GAMEPAD_BINDTYPE_AXIS) {
                    value = SDL_GetJoystickAxis(gamepad->joystick, binding->input.axis.axis);
                    if (binding->input.axis.axis_min < binding->input.axis.axis_max) {
                        valid_input_range = (value >= binding->input.axis.axis_min && value <= binding->input.axis.axis_max);
                    } else {
                        valid_input_range = (value >= binding->input.axis.axis_max && value <= binding->input.axis.axis_min);
                    }
                    if (valid_input_range) {
                        if (binding->input.axis.axis_min != binding->output.axis.axis_min || binding->input.axis.axis_max != binding->output.axis.axis_max) {
                            float normalized_value = (float)(value - binding->input.axis.axis_min) / (binding->input.axis.axis_max - binding->input.axis.axis_min);
                            value = binding->output.axis.axis_min + (int)(normalized_value * (binding->output.axis.axis_max - binding->output.axis.axis_min));
                        }
                    } else {
                        value = 0;
                    }
                } else if (binding->input_type == SDL_GAMEPAD_BINDTYPE_BUTTON) {
                    if (SDL_GetJoystickButton(gamepad->joystick, binding->input.button)) {
                        value = binding->output.axis.axis_max;
                    }
                } else if (binding->input_type == SDL_GAMEPAD_BINDTYPE_HAT) {
                    int hat_mask = SDL_GetJoystickHat(gamepad->joystick, binding->input.hat.hat);
                    if (hat_mask & binding->input.hat.hat_mask) {
                        value = binding->output.axis.axis_max;
                    }
                }

                if (binding->output.axis.axis_min < binding->output.axis.axis_max) {
                    valid_output_range = (value >= binding->output.axis.axis_min && value <= binding->output.axis.axis_max);
                } else {
                    valid_output_range = (value >= binding->output.axis.axis_max && value <= binding->output.axis.axis_min);
                }
                // If the value is zero, there might be another binding that makes it non-zero
                if (value != 0 && valid_output_range) {
                    result = (Sint16)value;
                    break;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/**
 *  Return whether a gamepad has a given button
 */
bool SDL_GamepadHasButton(SDL_Gamepad *gamepad, SDL_GamepadButton button)
{
    bool result = false;

    SDL_LockJoysticks();
    {
        int i;

        CHECK_GAMEPAD_MAGIC(gamepad, false);

        for (i = 0; i < gamepad->num_bindings; ++i) {
            const SDL_GamepadBinding *binding = &gamepad->bindings[i];
            if (binding->output_type == SDL_GAMEPAD_BINDTYPE_BUTTON && binding->output.button == button) {
                result = true;
                break;
            }
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 * Get the current state of a button on a gamepad
 */
bool SDL_GetGamepadButton(SDL_Gamepad *gamepad, SDL_GamepadButton button)
{
    bool result = false;

    SDL_LockJoysticks();
    {
        int i;

        CHECK_GAMEPAD_MAGIC(gamepad, false);

        for (i = 0; i < gamepad->num_bindings; ++i) {
            const SDL_GamepadBinding *binding = &gamepad->bindings[i];
            if (binding->output_type == SDL_GAMEPAD_BINDTYPE_BUTTON && binding->output.button == button) {
                if (binding->input_type == SDL_GAMEPAD_BINDTYPE_AXIS) {
                    bool valid_input_range;

                    int value = SDL_GetJoystickAxis(gamepad->joystick, binding->input.axis.axis);
                    int threshold = binding->input.axis.axis_min + (binding->input.axis.axis_max - binding->input.axis.axis_min) / 2;
                    if (binding->input.axis.axis_min < binding->input.axis.axis_max) {
                        valid_input_range = (value >= binding->input.axis.axis_min && value <= binding->input.axis.axis_max);
                        if (valid_input_range) {
                            result |= (value >= threshold);
                        }
                    } else {
                        valid_input_range = (value >= binding->input.axis.axis_max && value <= binding->input.axis.axis_min);
                        if (valid_input_range) {
                            result |= (value <= threshold);
                        }
                    }
                } else if (binding->input_type == SDL_GAMEPAD_BINDTYPE_BUTTON) {
                    result |= SDL_GetJoystickButton(gamepad->joystick, binding->input.button);
                } else if (binding->input_type == SDL_GAMEPAD_BINDTYPE_HAT) {
                    int hat_mask = SDL_GetJoystickHat(gamepad->joystick, binding->input.hat.hat);
                    result |= ((hat_mask & binding->input.hat.hat_mask) != 0);
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/**
 * Get the label of a button on a gamepad.
 */
static SDL_GamepadButtonLabel SDL_GetGamepadButtonLabelForFaceStyle(SDL_GamepadFaceStyle face_style, SDL_GamepadButton button)
{
    SDL_GamepadButtonLabel label = SDL_GAMEPAD_BUTTON_LABEL_UNKNOWN;

    switch (face_style) {
    case SDL_GAMEPAD_FACE_STYLE_ABXY:
        switch (button) {
        case SDL_GAMEPAD_BUTTON_SOUTH:
            label = SDL_GAMEPAD_BUTTON_LABEL_A;
            break;
        case SDL_GAMEPAD_BUTTON_EAST:
            label = SDL_GAMEPAD_BUTTON_LABEL_B;
            break;
        case SDL_GAMEPAD_BUTTON_WEST:
            label = SDL_GAMEPAD_BUTTON_LABEL_X;
            break;
        case SDL_GAMEPAD_BUTTON_NORTH:
            label = SDL_GAMEPAD_BUTTON_LABEL_Y;
            break;
        default:
            break;
        }
        break;
    case SDL_GAMEPAD_FACE_STYLE_BAYX:
        switch (button) {
        case SDL_GAMEPAD_BUTTON_SOUTH:
            label = SDL_GAMEPAD_BUTTON_LABEL_B;
            break;
        case SDL_GAMEPAD_BUTTON_EAST:
            label = SDL_GAMEPAD_BUTTON_LABEL_A;
            break;
        case SDL_GAMEPAD_BUTTON_WEST:
            label = SDL_GAMEPAD_BUTTON_LABEL_Y;
            break;
        case SDL_GAMEPAD_BUTTON_NORTH:
            label = SDL_GAMEPAD_BUTTON_LABEL_X;
            break;
        default:
            break;
        }
        break;
    case SDL_GAMEPAD_FACE_STYLE_SONY:
        switch (button) {
        case SDL_GAMEPAD_BUTTON_SOUTH:
            label = SDL_GAMEPAD_BUTTON_LABEL_CROSS;
            break;
        case SDL_GAMEPAD_BUTTON_EAST:
            label = SDL_GAMEPAD_BUTTON_LABEL_CIRCLE;
            break;
        case SDL_GAMEPAD_BUTTON_WEST:
            label = SDL_GAMEPAD_BUTTON_LABEL_SQUARE;
            break;
        case SDL_GAMEPAD_BUTTON_NORTH:
            label = SDL_GAMEPAD_BUTTON_LABEL_TRIANGLE;
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
    return label;
}

/**
 * Get the label of a button on a gamepad.
 */
SDL_GamepadButtonLabel SDL_GetGamepadButtonLabelForType(SDL_GamepadType type, SDL_GamepadButton button)
{
    return SDL_GetGamepadButtonLabelForFaceStyle(SDL_GetGamepadFaceStyleForGamepadType(type), button);
}

/**
 * Get the label of a button on a gamepad.
 */
SDL_GamepadButtonLabel SDL_GetGamepadButtonLabel(SDL_Gamepad *gamepad, SDL_GamepadButton button)
{
    SDL_GamepadFaceStyle face_style;

    SDL_LockJoysticks();
    {
        CHECK_GAMEPAD_MAGIC(gamepad, SDL_GAMEPAD_BUTTON_LABEL_UNKNOWN);

        face_style = gamepad->face_style;
    }
    SDL_UnlockJoysticks();

    return SDL_GetGamepadButtonLabelForFaceStyle(face_style, button);
}

/**
 *  Get the number of touchpads on a gamepad.
 */
int SDL_GetNumGamepadTouchpads(SDL_Gamepad *gamepad)
{
    int result = 0;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);
        if (joystick) {
            result = joystick->ntouchpads;
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/**
 *  Get the number of supported simultaneous fingers on a touchpad on a gamepad.
 */
int SDL_GetNumGamepadTouchpadFingers(SDL_Gamepad *gamepad, int touchpad)
{
    int result = 0;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);
        if (joystick) {
            if (touchpad >= 0 && touchpad < joystick->ntouchpads) {
                result = joystick->touchpads[touchpad].nfingers;
            }
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/**
 *  Get the current state of a finger on a touchpad on a gamepad.
 */
bool SDL_GetGamepadTouchpadFinger(SDL_Gamepad *gamepad, int touchpad, int finger, bool *down, float *x, float *y, float *pressure)
{
    bool result = false;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);
        if (joystick) {
            if (touchpad >= 0 && touchpad < joystick->ntouchpads) {
                SDL_JoystickTouchpadInfo *touchpad_info = &joystick->touchpads[touchpad];
                if (finger >= 0 && finger < touchpad_info->nfingers) {
                    SDL_JoystickTouchpadFingerInfo *info = &touchpad_info->fingers[finger];

                    if (down) {
                        *down = info->down;
                    }
                    if (x) {
                        *x = info->x;
                    }
                    if (y) {
                        *y = info->y;
                    }
                    if (pressure) {
                        *pressure = info->pressure;
                    }
                    result = true;
                } else {
                    result = SDL_InvalidParamError("finger");
                }
            } else {
                result = SDL_InvalidParamError("touchpad");
            }
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/**
 *  Return whether a gamepad has a particular sensor.
 */
bool SDL_GamepadHasSensor(SDL_Gamepad *gamepad, SDL_SensorType type)
{
    bool result = false;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);
        if (joystick) {
            int i;
            for (i = 0; i < joystick->nsensors; ++i) {
                if (joystick->sensors[i].type == type) {
                    result = true;
                    break;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 *  Set whether data reporting for a gamepad sensor is enabled
 */
bool SDL_SetGamepadSensorEnabled(SDL_Gamepad *gamepad, SDL_SensorType type, bool enabled)
{
    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);
        if (joystick) {
            int i;
            for (i = 0; i < joystick->nsensors; ++i) {
                SDL_JoystickSensorInfo *sensor = &joystick->sensors[i];

                if (sensor->type == type) {
                    if (sensor->enabled == (enabled != false)) {
                        SDL_UnlockJoysticks();
                        return true;
                    }

                    if (type == SDL_SENSOR_ACCEL && joystick->accel_sensor) {
                        if (enabled) {
                            joystick->accel = SDL_OpenSensor(joystick->accel_sensor);
                            if (!joystick->accel) {
                                SDL_UnlockJoysticks();
                                return false;
                            }
                        } else {
                            if (joystick->accel) {
                                SDL_CloseSensor(joystick->accel);
                                joystick->accel = NULL;
                            }
                        }
                    } else if (type == SDL_SENSOR_GYRO && joystick->gyro_sensor) {
                        if (enabled) {
                            joystick->gyro = SDL_OpenSensor(joystick->gyro_sensor);
                            if (!joystick->gyro) {
                                SDL_UnlockJoysticks();
                                return false;
                            }
                        } else {
                            if (joystick->gyro) {
                                SDL_CloseSensor(joystick->gyro);
                                joystick->gyro = NULL;
                            }
                        }
                    } else {
                        if (enabled) {
                            if (joystick->nsensors_enabled == 0) {
                                if (!joystick->driver->SetSensorsEnabled(joystick, true)) {
                                    SDL_UnlockJoysticks();
                                    return false;
                                }
                            }
                            ++joystick->nsensors_enabled;
                        } else {
                            if (joystick->nsensors_enabled == 1) {
                                if (!joystick->driver->SetSensorsEnabled(joystick, false)) {
                                    SDL_UnlockJoysticks();
                                    return false;
                                }
                            }
                            --joystick->nsensors_enabled;
                        }
                    }

                    sensor->enabled = enabled;
                    SDL_UnlockJoysticks();
                    return true;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return SDL_Unsupported();
}

/*
 *  Query whether sensor data reporting is enabled for a gamepad
 */
bool SDL_GamepadSensorEnabled(SDL_Gamepad *gamepad, SDL_SensorType type)
{
    bool result = false;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);
        if (joystick) {
            int i;
            for (i = 0; i < joystick->nsensors; ++i) {
                if (joystick->sensors[i].type == type) {
                    result = joystick->sensors[i].enabled;
                    break;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 *  Get the data rate of a gamepad sensor.
 */
float SDL_GetGamepadSensorDataRate(SDL_Gamepad *gamepad, SDL_SensorType type)
{
    float result = 0.0f;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);
        if (joystick) {
            int i;
            for (i = 0; i < joystick->nsensors; ++i) {
                SDL_JoystickSensorInfo *sensor = &joystick->sensors[i];

                if (sensor->type == type) {
                    result = sensor->rate;
                    break;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 *  Get the current state of a gamepad sensor.
 */
bool SDL_GetGamepadSensorData(SDL_Gamepad *gamepad, SDL_SensorType type, float *data, int num_values)
{
    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);
        if (joystick) {
            int i;
            for (i = 0; i < joystick->nsensors; ++i) {
                SDL_JoystickSensorInfo *sensor = &joystick->sensors[i];

                if (sensor->type == type) {
                    num_values = SDL_min(num_values, SDL_arraysize(sensor->data));
                    SDL_memcpy(data, sensor->data, num_values * sizeof(*data));
                    SDL_UnlockJoysticks();
                    return true;
                }
            }
        }
    }
    SDL_UnlockJoysticks();

    return SDL_Unsupported();
}

SDL_JoystickID SDL_GetGamepadID(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return 0;
    }
    return SDL_GetJoystickID(joystick);
}

SDL_PropertiesID SDL_GetGamepadProperties(SDL_Gamepad *gamepad)
{
    SDL_PropertiesID result = 0;

    SDL_LockJoysticks();
    {
        CHECK_GAMEPAD_MAGIC(gamepad, 0);

        result = SDL_GetJoystickProperties(gamepad->joystick);
    }
    SDL_UnlockJoysticks();

    return result;
}

const char *SDL_GetGamepadName(SDL_Gamepad *gamepad)
{
    const char *result = NULL;

    SDL_LockJoysticks();
    {
        CHECK_GAMEPAD_MAGIC(gamepad, NULL);

        if (SDL_strcmp(gamepad->name, "*") == 0 ||
            gamepad->joystick->steam_handle != 0) {
            result = SDL_GetJoystickName(gamepad->joystick);
        } else {
            result = SDL_GetPersistentString(gamepad->name);
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

const char *SDL_GetGamepadPath(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return NULL;
    }
    return SDL_GetJoystickPath(joystick);
}

SDL_GamepadType SDL_GetGamepadType(SDL_Gamepad *gamepad)
{
    SDL_GamepadType type;
    const SDL_SteamVirtualGamepadInfo *info;

    SDL_LockJoysticks();
    {
        CHECK_GAMEPAD_MAGIC(gamepad, SDL_GAMEPAD_TYPE_UNKNOWN);

        info = SDL_GetJoystickVirtualGamepadInfoForID(gamepad->joystick->instance_id);
        if (info) {
            type = info->type;
        } else {
            type = gamepad->type;
        }
    }
    SDL_UnlockJoysticks();

    return type;
}

SDL_GamepadType SDL_GetRealGamepadType(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return SDL_GAMEPAD_TYPE_UNKNOWN;
    }
    return SDL_GetGamepadTypeFromGUID(SDL_GetJoystickGUID(joystick), SDL_GetJoystickName(joystick));
}

int SDL_GetGamepadPlayerIndex(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return -1;
    }
    return SDL_GetJoystickPlayerIndex(joystick);
}

/**
 *  Set the player index of an opened gamepad
 */
bool SDL_SetGamepadPlayerIndex(SDL_Gamepad *gamepad, int player_index)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        // SDL_SetError() will have been called already by SDL_GetGamepadJoystick()
        return false;
    }
    return SDL_SetJoystickPlayerIndex(joystick, player_index);
}

Uint16 SDL_GetGamepadVendor(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return 0;
    }
    return SDL_GetJoystickVendor(joystick);
}

Uint16 SDL_GetGamepadProduct(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return 0;
    }
    return SDL_GetJoystickProduct(joystick);
}

Uint16 SDL_GetGamepadProductVersion(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return 0;
    }
    return SDL_GetJoystickProductVersion(joystick);
}

Uint16 SDL_GetGamepadFirmwareVersion(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return 0;
    }
    return SDL_GetJoystickFirmwareVersion(joystick);
}

const char * SDL_GetGamepadSerial(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return NULL;
    }
    return SDL_GetJoystickSerial(joystick);

}

Uint64 SDL_GetGamepadSteamHandle(SDL_Gamepad *gamepad)
{
    Uint64 handle = 0;

    SDL_LockJoysticks();
    {
        CHECK_GAMEPAD_MAGIC(gamepad, 0);

        handle = gamepad->joystick->steam_handle;
    }
    SDL_UnlockJoysticks();

    return handle;
}

SDL_JoystickConnectionState SDL_GetGamepadConnectionState(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return SDL_JOYSTICK_CONNECTION_INVALID;
    }
    return SDL_GetJoystickConnectionState(joystick);
}

SDL_PowerState SDL_GetGamepadPowerInfo(SDL_Gamepad *gamepad, int *percent)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (percent) {
        *percent = -1;
    }
    if (!joystick) {
        return SDL_POWERSTATE_ERROR;
    }
    return SDL_GetJoystickPowerInfo(joystick, percent);
}

/*
 * Return if the gamepad in question is currently attached to the system,
 *  \return 0 if not plugged in, 1 if still present.
 */
bool SDL_GamepadConnected(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return false;
    }
    return SDL_JoystickConnected(joystick);
}

/*
 * Get the joystick for this gamepad
 */
SDL_Joystick *SDL_GetGamepadJoystick(SDL_Gamepad *gamepad)
{
    SDL_Joystick *joystick;

    SDL_LockJoysticks();
    {
        CHECK_GAMEPAD_MAGIC(gamepad, NULL);

        joystick = gamepad->joystick;
    }
    SDL_UnlockJoysticks();

    return joystick;
}

/*
 * Return the SDL_Gamepad associated with an instance id.
 */
SDL_Gamepad *SDL_GetGamepadFromID(SDL_JoystickID joyid)
{
    SDL_Gamepad *gamepad;

    SDL_LockJoysticks();
    gamepad = SDL_gamepads;
    while (gamepad) {
        if (gamepad->joystick->instance_id == joyid) {
            SDL_UnlockJoysticks();
            return gamepad;
        }
        gamepad = gamepad->next;
    }
    SDL_UnlockJoysticks();
    return NULL;
}

/**
 * Return the SDL_Gamepad associated with a player index.
 */
SDL_Gamepad *SDL_GetGamepadFromPlayerIndex(int player_index)
{
    SDL_Gamepad *result = NULL;

    SDL_LockJoysticks();
    {
        SDL_Joystick *joystick = SDL_GetJoystickFromPlayerIndex(player_index);
        if (joystick) {
            result = SDL_GetGamepadFromID(joystick->instance_id);
        }
    }
    SDL_UnlockJoysticks();

    return result;
}

/*
 * Get the SDL joystick layer bindings for this gamepad
 */
SDL_GamepadBinding **SDL_GetGamepadBindings(SDL_Gamepad *gamepad, int *count)
{
    SDL_GamepadBinding **bindings = NULL;

    if (count) {
        *count = 0;
    }

    SDL_LockJoysticks();
    {
        CHECK_GAMEPAD_MAGIC(gamepad, NULL);

        size_t pointers_size = ((gamepad->num_bindings + 1) * sizeof(SDL_GamepadBinding *));
        size_t elements_size = (gamepad->num_bindings * sizeof(SDL_GamepadBinding));
        bindings = (SDL_GamepadBinding **)SDL_malloc(pointers_size + elements_size);
        if (bindings) {
            SDL_GamepadBinding *binding = (SDL_GamepadBinding *)((Uint8 *)bindings + pointers_size);
            int i;
            for (i = 0; i < gamepad->num_bindings; ++i, ++binding) {
                bindings[i] = binding;
                SDL_copyp(binding, &gamepad->bindings[i]);
            }
            bindings[i] = NULL;

            if (count) {
                *count = gamepad->num_bindings;
            }
        }
    }
    SDL_UnlockJoysticks();

    return bindings;
}

bool SDL_RumbleGamepad(SDL_Gamepad *gamepad, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble, Uint32 duration_ms)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return false;
    }
    return SDL_RumbleJoystick(joystick, low_frequency_rumble, high_frequency_rumble, duration_ms);
}

bool SDL_RumbleGamepadTriggers(SDL_Gamepad *gamepad, Uint16 left_rumble, Uint16 right_rumble, Uint32 duration_ms)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return false;
    }
    return SDL_RumbleJoystickTriggers(joystick, left_rumble, right_rumble, duration_ms);
}

bool SDL_SetGamepadLED(SDL_Gamepad *gamepad, Uint8 red, Uint8 green, Uint8 blue)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return false;
    }
    return SDL_SetJoystickLED(joystick, red, green, blue);
}

bool SDL_SendGamepadEffect(SDL_Gamepad *gamepad, const void *data, int size)
{
    SDL_Joystick *joystick = SDL_GetGamepadJoystick(gamepad);

    if (!joystick) {
        return false;
    }
    return SDL_SendJoystickEffect(joystick, data, size);
}

void SDL_CloseGamepad(SDL_Gamepad *gamepad)
{
    SDL_Gamepad *gamepadlist, *gamepadlistprev;

    SDL_LockJoysticks();

    if (!SDL_ObjectValid(gamepad, SDL_OBJECT_TYPE_GAMEPAD)) {
        SDL_UnlockJoysticks();
        return;
    }

    // First decrement ref count
    if (--gamepad->ref_count > 0) {
        SDL_UnlockJoysticks();
        return;
    }

    SDL_CloseJoystick(gamepad->joystick);

    gamepadlist = SDL_gamepads;
    gamepadlistprev = NULL;
    while (gamepadlist) {
        if (gamepad == gamepadlist) {
            if (gamepadlistprev) {
                // unlink this entry
                gamepadlistprev->next = gamepadlist->next;
            } else {
                SDL_gamepads = gamepad->next;
            }
            break;
        }
        gamepadlistprev = gamepadlist;
        gamepadlist = gamepadlist->next;
    }

    SDL_SetObjectValid(gamepad, SDL_OBJECT_TYPE_GAMEPAD, false);
    SDL_free(gamepad->bindings);
    SDL_free(gamepad->last_match_axis);
    SDL_free(gamepad->last_hat_mask);
    SDL_free(gamepad);

    SDL_UnlockJoysticks();
}

/*
 * Quit the gamepad subsystem
 */
void SDL_QuitGamepads(void)
{
    SDL_Gamepad *gamepad;

    SDL_LockJoysticks();

    for (gamepad = SDL_gamepads; gamepad; gamepad = gamepad->next) {
        SDL_PrivateGamepadRemoved(gamepad->joystick->instance_id);
    }

    SDL_gamepads_initialized = false;

    SDL_RemoveEventWatch(SDL_GamepadEventWatcher, NULL);

    while (SDL_gamepads) {
        SDL_gamepads->ref_count = 1;
        SDL_CloseGamepad(SDL_gamepads);
    }

    SDL_UnlockJoysticks();
}

void SDL_QuitGamepadMappings(void)
{
    GamepadMapping_t *pGamepadMap;

    SDL_AssertJoysticksLocked();

    while (s_pSupportedGamepads) {
        pGamepadMap = s_pSupportedGamepads;
        s_pSupportedGamepads = s_pSupportedGamepads->next;
        SDL_free(pGamepadMap->name);
        SDL_free(pGamepadMap->mapping);
        SDL_free(pGamepadMap);
    }

    SDL_FreeVIDPIDList(&SDL_allowed_gamepads);
    SDL_FreeVIDPIDList(&SDL_ignored_gamepads);

    if (s_gamepadInstanceIDs) {
        SDL_DestroyHashTable(s_gamepadInstanceIDs);
        s_gamepadInstanceIDs = NULL;
    }
}

/*
 * Event filter to transform joystick events into appropriate gamepad ones
 */
static void SDL_SendGamepadAxis(Uint64 timestamp, SDL_Gamepad *gamepad, SDL_GamepadAxis axis, Sint16 value)
{
    SDL_AssertJoysticksLocked();

    // translate the event, if desired
    if (SDL_EventEnabled(SDL_EVENT_GAMEPAD_AXIS_MOTION)) {
        SDL_Event event;
        event.type = SDL_EVENT_GAMEPAD_AXIS_MOTION;
        event.common.timestamp = timestamp;
        event.gaxis.which = gamepad->joystick->instance_id;
        event.gaxis.axis = axis;
        event.gaxis.value = value;
        SDL_PushEvent(&event);
    }
}

static void SDL_SendGamepadButton(Uint64 timestamp, SDL_Gamepad *gamepad, SDL_GamepadButton button, bool down)
{
    SDL_Event event;

    SDL_AssertJoysticksLocked();

    if (button == SDL_GAMEPAD_BUTTON_INVALID) {
        return;
    }

    if (down) {
        event.type = SDL_EVENT_GAMEPAD_BUTTON_DOWN;
    } else {
        event.type = SDL_EVENT_GAMEPAD_BUTTON_UP;
    }

    if (button == SDL_GAMEPAD_BUTTON_GUIDE) {
        Uint64 now = SDL_GetTicks();
        if (down) {
            gamepad->guide_button_down = now;

            if (gamepad->joystick->delayed_guide_button) {
                // Skip duplicate press
                return;
            }
        } else {
            if (now < (gamepad->guide_button_down + SDL_MINIMUM_GUIDE_BUTTON_DELAY_MS)) {
                gamepad->joystick->delayed_guide_button = true;
                return;
            }
            gamepad->joystick->delayed_guide_button = false;
        }
    }

    // translate the event, if desired
    if (SDL_EventEnabled(event.type)) {
        event.common.timestamp = timestamp;
        event.gbutton.which = gamepad->joystick->instance_id;
        event.gbutton.button = button;
        event.gbutton.down = down;
        SDL_PushEvent(&event);
    }
}

static const Uint32 SDL_gamepad_event_list[] = {
    SDL_EVENT_GAMEPAD_AXIS_MOTION,
    SDL_EVENT_GAMEPAD_BUTTON_DOWN,
    SDL_EVENT_GAMEPAD_BUTTON_UP,
    SDL_EVENT_GAMEPAD_ADDED,
    SDL_EVENT_GAMEPAD_REMOVED,
    SDL_EVENT_GAMEPAD_REMAPPED,
    SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN,
    SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION,
    SDL_EVENT_GAMEPAD_TOUCHPAD_UP,
    SDL_EVENT_GAMEPAD_SENSOR_UPDATE,
};

void SDL_SetGamepadEventsEnabled(bool enabled)
{
    unsigned int i;

    for (i = 0; i < SDL_arraysize(SDL_gamepad_event_list); ++i) {
        SDL_SetEventEnabled(SDL_gamepad_event_list[i], enabled);
    }
}

bool SDL_GamepadEventsEnabled(void)
{
    bool enabled = false;
    unsigned int i;

    for (i = 0; i < SDL_arraysize(SDL_gamepad_event_list); ++i) {
        enabled = SDL_EventEnabled(SDL_gamepad_event_list[i]);
        if (enabled) {
            break;
        }
    }
    return enabled;
}

void SDL_GamepadHandleDelayedGuideButton(SDL_Joystick *joystick)
{
    SDL_Gamepad *gamepad;

    SDL_AssertJoysticksLocked();

    for (gamepad = SDL_gamepads; gamepad; gamepad = gamepad->next) {
        if (gamepad->joystick == joystick) {
            SDL_SendGamepadButton(0, gamepad, SDL_GAMEPAD_BUTTON_GUIDE, false);

            // Make sure we send an update complete event for this change
            if (!gamepad->joystick->update_complete) {
                gamepad->joystick->update_complete = SDL_GetTicksNS();
            }
            break;
        }
    }
}

const char *SDL_GetGamepadAppleSFSymbolsNameForButton(SDL_Gamepad *gamepad, SDL_GamepadButton button)
{
    const char *result = NULL;
#ifdef SDL_JOYSTICK_MFI
    const char *IOS_GetAppleSFSymbolsNameForButton(SDL_Gamepad *gamepad, SDL_GamepadButton button);

    SDL_LockJoysticks();
    {
        CHECK_GAMEPAD_MAGIC(gamepad, NULL);

        result = IOS_GetAppleSFSymbolsNameForButton(gamepad, button);
    }
    SDL_UnlockJoysticks();
#endif
    return result;
}

const char *SDL_GetGamepadAppleSFSymbolsNameForAxis(SDL_Gamepad *gamepad, SDL_GamepadAxis axis)
{
    const char *result = NULL;
#ifdef SDL_JOYSTICK_MFI
    const char *IOS_GetAppleSFSymbolsNameForAxis(SDL_Gamepad *gamepad, SDL_GamepadAxis axis);

    SDL_LockJoysticks();
    {
        CHECK_GAMEPAD_MAGIC(gamepad, NULL);

        result = IOS_GetAppleSFSymbolsNameForAxis(gamepad, axis);
    }
    SDL_UnlockJoysticks();
#endif
    return result;
}
