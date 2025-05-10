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

#ifdef SDL_JOYSTICK_EMSCRIPTEN

#include <stdio.h> // For the definition of NULL

#include "SDL_sysjoystick_c.h"
#include "../SDL_joystick_c.h"

static SDL_joylist_item *JoystickByIndex(int index);

static SDL_joylist_item *SDL_joylist = NULL;
static SDL_joylist_item *SDL_joylist_tail = NULL;
static int numjoysticks = 0;

static EM_BOOL Emscripten_JoyStickConnected(int eventType, const EmscriptenGamepadEvent *gamepadEvent, void *userData)
{
    SDL_joylist_item *item;
    int i;

    SDL_LockJoysticks();

    if (JoystickByIndex(gamepadEvent->index) != NULL) {
        goto done;
    }

    item = (SDL_joylist_item *)SDL_malloc(sizeof(SDL_joylist_item));
    if (!item) {
        goto done;
    }

    SDL_zerop(item);
    item->index = gamepadEvent->index;

    item->name = SDL_CreateJoystickName(0, 0, NULL, gamepadEvent->id);
    if (!item->name) {
        SDL_free(item);
        goto done;
    }

    item->mapping = SDL_strdup(gamepadEvent->mapping);
    if (!item->mapping) {
        SDL_free(item->name);
        SDL_free(item);
        goto done;
    }

    item->naxes = gamepadEvent->numAxes;
    item->nbuttons = gamepadEvent->numButtons;
    item->device_instance = SDL_GetNextObjectID();

    item->timestamp = gamepadEvent->timestamp;

    for (i = 0; i < item->naxes; i++) {
        item->axis[i] = gamepadEvent->axis[i];
    }

    for (i = 0; i < item->nbuttons; i++) {
        item->analogButton[i] = gamepadEvent->analogButton[i];
        item->digitalButton[i] = gamepadEvent->digitalButton[i];
    }

    if (!SDL_joylist_tail) {
        SDL_joylist = SDL_joylist_tail = item;
    } else {
        SDL_joylist_tail->next = item;
        SDL_joylist_tail = item;
    }

    ++numjoysticks;

    SDL_PrivateJoystickAdded(item->device_instance);

#ifdef DEBUG_JOYSTICK
    SDL_Log("Number of joysticks is %d", numjoysticks);
#endif
#ifdef DEBUG_JOYSTICK
    SDL_Log("Added joystick with index %d", item->index);
#endif

done:
    SDL_UnlockJoysticks();

    return 1;
}

static EM_BOOL Emscripten_JoyStickDisconnected(int eventType, const EmscriptenGamepadEvent *gamepadEvent, void *userData)
{
    SDL_joylist_item *item = SDL_joylist;
    SDL_joylist_item *prev = NULL;

    SDL_LockJoysticks();

    while (item) {
        if (item->index == gamepadEvent->index) {
            break;
        }
        prev = item;
        item = item->next;
    }

    if (!item) {
        goto done;
    }

    if (item->joystick) {
        item->joystick->hwdata = NULL;
    }

    if (prev) {
        prev->next = item->next;
    } else {
        SDL_assert(SDL_joylist == item);
        SDL_joylist = item->next;
    }
    if (item == SDL_joylist_tail) {
        SDL_joylist_tail = prev;
    }

    // Need to decrement the joystick count before we post the event
    --numjoysticks;

    SDL_PrivateJoystickRemoved(item->device_instance);

#ifdef DEBUG_JOYSTICK
    SDL_Log("Removed joystick with id %d", item->device_instance);
#endif
    SDL_free(item->name);
    SDL_free(item->mapping);
    SDL_free(item);

done:
    SDL_UnlockJoysticks();

    return 1;
}

// Function to perform any system-specific joystick related cleanup
static void EMSCRIPTEN_JoystickQuit(void)
{
    SDL_joylist_item *item = NULL;
    SDL_joylist_item *next = NULL;

    for (item = SDL_joylist; item; item = next) {
        next = item->next;
        SDL_free(item->mapping);
        SDL_free(item->name);
        SDL_free(item);
    }

    SDL_joylist = SDL_joylist_tail = NULL;

    numjoysticks = 0;

    emscripten_set_gamepadconnected_callback(NULL, 0, NULL);
    emscripten_set_gamepaddisconnected_callback(NULL, 0, NULL);
}

// Function to scan the system for joysticks.
static bool EMSCRIPTEN_JoystickInit(void)
{
    int rc, i, numjs;
    EmscriptenGamepadEvent gamepadState;

    numjoysticks = 0;

    rc = emscripten_sample_gamepad_data();

    // Check if gamepad is supported by browser
    if (rc == EMSCRIPTEN_RESULT_NOT_SUPPORTED) {
        return SDL_SetError("Gamepads not supported");
    }

    numjs = emscripten_get_num_gamepads();

    // handle already connected gamepads
    if (numjs > 0) {
        for (i = 0; i < numjs; i++) {
            rc = emscripten_get_gamepad_status(i, &gamepadState);
            if (rc == EMSCRIPTEN_RESULT_SUCCESS) {
                Emscripten_JoyStickConnected(EMSCRIPTEN_EVENT_GAMEPADCONNECTED,
                                             &gamepadState,
                                             NULL);
            }
        }
    }

    rc = emscripten_set_gamepadconnected_callback(NULL,
                                                  0,
                                                  Emscripten_JoyStickConnected);

    if (rc != EMSCRIPTEN_RESULT_SUCCESS) {
        EMSCRIPTEN_JoystickQuit();
        return SDL_SetError("Could not set gamepad connect callback");
    }

    rc = emscripten_set_gamepaddisconnected_callback(NULL,
                                                         0,
                                                         Emscripten_JoyStickDisconnected);
    if (rc != EMSCRIPTEN_RESULT_SUCCESS) {
        EMSCRIPTEN_JoystickQuit();
        return SDL_SetError("Could not set gamepad disconnect callback");
    }

    return true;
}

// Returns item matching given SDL device index.
static SDL_joylist_item *JoystickByDeviceIndex(int device_index)
{
    SDL_joylist_item *item = SDL_joylist;

    while (0 < device_index) {
        --device_index;
        item = item->next;
    }

    return item;
}

// Returns item matching given HTML gamepad index.
static SDL_joylist_item *JoystickByIndex(int index)
{
    SDL_joylist_item *item = SDL_joylist;

    if (index < 0) {
        return NULL;
    }

    while (item) {
        if (item->index == index) {
            break;
        }
        item = item->next;
    }

    return item;
}

static int EMSCRIPTEN_JoystickGetCount(void)
{
    return numjoysticks;
}

static void EMSCRIPTEN_JoystickDetect(void)
{
}

static bool EMSCRIPTEN_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    // We don't override any other drivers
    return false;
}

static const char *EMSCRIPTEN_JoystickGetDeviceName(int device_index)
{
    return JoystickByDeviceIndex(device_index)->name;
}

static const char *EMSCRIPTEN_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int EMSCRIPTEN_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int EMSCRIPTEN_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void EMSCRIPTEN_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static SDL_JoystickID EMSCRIPTEN_JoystickGetDeviceInstanceID(int device_index)
{
    return JoystickByDeviceIndex(device_index)->device_instance;
}

static bool EMSCRIPTEN_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    SDL_joylist_item *item = JoystickByDeviceIndex(device_index);

    if (!item) {
        return SDL_SetError("No such device");
    }

    if (item->joystick) {
        return SDL_SetError("Joystick already opened");
    }

    joystick->hwdata = (struct joystick_hwdata *)item;
    item->joystick = joystick;

    // HTML5 Gamepad API doesn't say anything about these
    joystick->nhats = 0;

    joystick->nbuttons = item->nbuttons;
    joystick->naxes = item->naxes;

    return true;
}

/* Function to update the state of a joystick - called as a device poll.
 * This function shouldn't update the joystick structure directly,
 * but instead should call SDL_PrivateJoystick*() to deliver events
 * and update joystick device state.
 */
static void EMSCRIPTEN_JoystickUpdate(SDL_Joystick *joystick)
{
    EmscriptenGamepadEvent gamepadState;
    SDL_joylist_item *item = (SDL_joylist_item *)joystick->hwdata;
    int i, result;
    Uint64 timestamp = SDL_GetTicksNS();

    emscripten_sample_gamepad_data();

    if (item) {
        result = emscripten_get_gamepad_status(item->index, &gamepadState);
        if (result == EMSCRIPTEN_RESULT_SUCCESS) {
            if (gamepadState.timestamp == 0 || gamepadState.timestamp != item->timestamp) {
                for (i = 0; i < item->nbuttons; i++) {
                    if (item->digitalButton[i] != gamepadState.digitalButton[i]) {
                        bool down = (gamepadState.digitalButton[i] != 0);
                        SDL_SendJoystickButton(timestamp, item->joystick, i, down);
                    }

                    // store values to compare them in the next update
                    item->analogButton[i] = gamepadState.analogButton[i];
                    item->digitalButton[i] = gamepadState.digitalButton[i];
                }

                for (i = 0; i < item->naxes; i++) {
                    if (item->axis[i] != gamepadState.axis[i]) {
                        // do we need to do conversion?
                        SDL_SendJoystickAxis(timestamp, item->joystick, i,
                                                (Sint16)(32767. * gamepadState.axis[i]));
                    }

                    // store to compare in next update
                    item->axis[i] = gamepadState.axis[i];
                }

                item->timestamp = gamepadState.timestamp;
            }
        }
    }
}

// Function to close a joystick after use
static void EMSCRIPTEN_JoystickClose(SDL_Joystick *joystick)
{
    SDL_joylist_item *item = (SDL_joylist_item *)joystick->hwdata;
    if (item) {
        item->joystick = NULL;
    }
}

static SDL_GUID EMSCRIPTEN_JoystickGetDeviceGUID(int device_index)
{
    // the GUID is just the name for now
    const char *name = EMSCRIPTEN_JoystickGetDeviceName(device_index);
    return SDL_CreateJoystickGUIDForName(name);
}

static bool EMSCRIPTEN_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    return SDL_Unsupported();
}

static bool EMSCRIPTEN_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static bool EMSCRIPTEN_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return false;
}

static bool EMSCRIPTEN_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool EMSCRIPTEN_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool EMSCRIPTEN_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

SDL_JoystickDriver SDL_EMSCRIPTEN_JoystickDriver = {
    EMSCRIPTEN_JoystickInit,
    EMSCRIPTEN_JoystickGetCount,
    EMSCRIPTEN_JoystickDetect,
    EMSCRIPTEN_JoystickIsDevicePresent,
    EMSCRIPTEN_JoystickGetDeviceName,
    EMSCRIPTEN_JoystickGetDevicePath,
    EMSCRIPTEN_JoystickGetDeviceSteamVirtualGamepadSlot,
    EMSCRIPTEN_JoystickGetDevicePlayerIndex,
    EMSCRIPTEN_JoystickSetDevicePlayerIndex,
    EMSCRIPTEN_JoystickGetDeviceGUID,
    EMSCRIPTEN_JoystickGetDeviceInstanceID,
    EMSCRIPTEN_JoystickOpen,
    EMSCRIPTEN_JoystickRumble,
    EMSCRIPTEN_JoystickRumbleTriggers,
    EMSCRIPTEN_JoystickSetLED,
    EMSCRIPTEN_JoystickSendEffect,
    EMSCRIPTEN_JoystickSetSensorsEnabled,
    EMSCRIPTEN_JoystickUpdate,
    EMSCRIPTEN_JoystickClose,
    EMSCRIPTEN_JoystickQuit,
    EMSCRIPTEN_JoystickGetGamepadMapping
};

#endif // SDL_JOYSTICK_EMSCRIPTEN
