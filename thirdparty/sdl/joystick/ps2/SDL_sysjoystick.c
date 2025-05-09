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

#ifdef SDL_JOYSTICK_PS2

// This is the PS2 implementation of the SDL joystick API
#include <libmtap.h>
#include <libpad.h>
#include <ps2_joystick_driver.h>

#include <stdio.h> // For the definition of NULL
#include <stdlib.h>

#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"

#define PS2_MAX_PORT      2 // each ps2 has 2 ports
#define PS2_MAX_SLOT      4 // maximum - 4 slots in one multitap
#define MAX_CONTROLLERS   (PS2_MAX_PORT * PS2_MAX_SLOT)
#define PS2_ANALOG_STICKS 2
#define PS2_ANALOG_AXIS   2
#define PS2_BUTTONS       16
#define PS2_TOTAL_AXIS    (PS2_ANALOG_STICKS * PS2_ANALOG_AXIS)

struct JoyInfo
{
    uint8_t padBuf[256];
    uint16_t btns;
    uint8_t analog_state[PS2_TOTAL_AXIS];
    uint8_t port;
    uint8_t slot;
    int8_t rumble_ready;
    int8_t opened;
} __attribute__((aligned(64)));

static uint8_t enabled_pads = 0;
static struct JoyInfo joyInfo[MAX_CONTROLLERS];

static inline int16_t convert_u8_to_s16(uint8_t val)
{
    if (val == 0) {
        return -0x7fff;
    }
    return val * 0x0101 - 0x8000;
}

static inline uint8_t rumble_status(uint8_t index)
{
    char actAlign[6];
    int res;
    struct JoyInfo *info = &joyInfo[index];

    if (info->rumble_ready == 0) {
        actAlign[0] = 0;
        actAlign[1] = 1;
        actAlign[2] = 0xff;
        actAlign[3] = 0xff;
        actAlign[4] = 0xff;
        actAlign[5] = 0xff;

        res = padSetActAlign(info->port, info->slot, actAlign);
        info->rumble_ready = res <= 0 ? -1 : 1;
    }

    return info->rumble_ready == 1;
}

// Function to scan the system for joysticks.
static bool PS2_JoystickInit(void)
{
    uint32_t port = 0;
    uint32_t slot = 0;

    if (init_joystick_driver(true) < 0) {
        return false;
    }

    for (port = 0; port < PS2_MAX_PORT; port++) {
        mtapPortOpen(port);
    }
    // it can fail - we dont care, we will check it more strictly when padPortOpen

    for (slot = 0; slot < PS2_MAX_SLOT; slot++) {
        for (port = 0; port < PS2_MAX_PORT; port++) {
            /* 2 main controller ports acts the same with and without multitap
            Port 0,0 -> Connector 1 - the same as Port 0
            Port 1,0 -> Connector 2 - the same as Port 1
            Port 0,1 -> Connector 3
            Port 1,1 -> Connector 4
            Port 0,2 -> Connector 5
            Port 1,2 -> Connector 6
            Port 0,3 -> Connector 7
            Port 1,3 -> Connector 8
            */

            struct JoyInfo *info = &joyInfo[enabled_pads];
            if (padPortOpen(port, slot, (void *)info->padBuf) > 0) {
                info->port = (uint8_t)port;
                info->slot = (uint8_t)slot;
                info->opened = 1;
                enabled_pads++;
                SDL_PrivateJoystickAdded(enabled_pads);
            }
        }
    }

    return (enabled_pads > 0);
}

// Function to return the number of joystick devices plugged in right now
static int PS2_JoystickGetCount(void)
{
    return (int)enabled_pads;
}

// Function to cause any queued joystick insertions to be processed
static void PS2_JoystickDetect(void)
{
}

static bool PS2_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    // We don't override any other drivers
    return false;
}

// Function to get the device-dependent name of a joystick
static const char *PS2_JoystickGetDeviceName(int index)
{
    if (index >= 0 && index < enabled_pads) {
        return "PS2 Controller";
    }

    SDL_SetError("No joystick available with that index");
    return NULL;
}

// Function to get the device-dependent path of a joystick
static const char *PS2_JoystickGetDevicePath(int index)
{
    return NULL;
}

// Function to get the Steam virtual gamepad slot of a joystick
static int PS2_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

// Function to get the player index of a joystick
static int PS2_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

// Function to set the player index of a joystick
static void PS2_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

// Function to return the stable GUID for a plugged in device
static SDL_GUID PS2_JoystickGetDeviceGUID(int device_index)
{
    // the GUID is just the name for now
    const char *name = PS2_JoystickGetDeviceName(device_index);
    return SDL_CreateJoystickGUIDForName(name);
}

// Function to get the current instance id of the joystick located at device_index
static SDL_JoystickID PS2_JoystickGetDeviceInstanceID(int device_index)
{
    return device_index + 1;
}

/*  Function to open a joystick for use.
    The joystick to open is specified by the device index.
    This should fill the nbuttons and naxes fields of the joystick structure.
    It returns 0, or -1 if there is an error.
*/
static bool PS2_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    int index = joystick->instance_id;
    struct JoyInfo *info = &joyInfo[index];

    if (!info->opened) {
        if (padPortOpen(info->port, info->slot, (void *)info->padBuf) > 0) {
            info->opened = 1;
        } else {
            return false;
        }
    }
    joystick->nbuttons = PS2_BUTTONS;
    joystick->naxes = PS2_TOTAL_AXIS;
    joystick->nhats = 0;

    SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_RUMBLE_BOOLEAN, true);

    return true;
}

// Rumble functionality
static bool PS2_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    char actAlign[6];
    int res;
    int index = joystick->instance_id;
    struct JoyInfo *info = &joyInfo[index];

    if (!rumble_status(index)) {
        return false;
    }

    // Initial value
    actAlign[0] = low_frequency_rumble >> 8;  // Enable small engine
    actAlign[1] = high_frequency_rumble >> 8; // Enable big engine
    actAlign[2] = 0xff;
    actAlign[3] = 0xff;
    actAlign[4] = 0xff;
    actAlign[5] = 0xff;

    res = padSetActDirect(info->port, info->slot, actAlign);
    return (res == 1);
}

// Rumble functionality
static bool PS2_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left, Uint16 right)
{
    return SDL_Unsupported();
}

// LED functionality
static bool PS2_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

// General effects
static bool PS2_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

// Sensor functionality
static bool PS2_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

/*  Function to update the state of a joystick - called as a device poll.
 *   This function shouldn't update the joystick structure directly,
 *   but instead should call SDL_PrivateJoystick*() to deliver events
 *   and update joystick device state.
 */
static void PS2_JoystickUpdate(SDL_Joystick *joystick)
{
    uint8_t i;
    uint8_t previous_axis, current_axis;
    uint16_t mask, previous, current;
    struct padButtonStatus buttons;
    uint8_t all_axis[PS2_TOTAL_AXIS];
    int index = joystick->instance_id;
    struct JoyInfo *info = &joyInfo[index];
    int state = padGetState(info->port, info->slot);
    Uint64 timestamp = SDL_GetTicksNS();

    if (state != PAD_STATE_DISCONN && state != PAD_STATE_EXECCMD && state != PAD_STATE_ERROR) {
        int ret = padRead(info->port, info->slot, &buttons); // port, slot, buttons
        if (ret != 0) {
            // Buttons
            int32_t pressed_buttons = 0xffff ^ buttons.btns;
            ;
            if (info->btns != pressed_buttons) {
                for (i = 0; i < PS2_BUTTONS; i++) {
                    mask = (1 << i);
                    previous = info->btns & mask;
                    current = pressed_buttons & mask;
                    if (previous != current) {
                        SDL_SendJoystickButton(timestamp, joystick, i, (current != 0));
                    }
                }
            }
            info->btns = pressed_buttons;

            // Analog
            all_axis[0] = buttons.ljoy_h;
            all_axis[1] = buttons.ljoy_v;
            all_axis[2] = buttons.rjoy_h;
            all_axis[3] = buttons.rjoy_v;

            for (i = 0; i < PS2_TOTAL_AXIS; i++) {
                previous_axis = info->analog_state[i];
                current_axis = all_axis[i];
                if (previous_axis != current_axis) {
                    SDL_SendJoystickAxis(timestamp, joystick, i, convert_u8_to_s16(current_axis));
                }

                info->analog_state[i] = current_axis;
            }
        }
    }
}

// Function to close a joystick after use
static void PS2_JoystickClose(SDL_Joystick *joystick)
{
    int index = joystick->instance_id;
    struct JoyInfo *info = &joyInfo[index];
    padPortClose(info->port, info->slot);
    info->opened = 0;
}

// Function to perform any system-specific joystick related cleanup
static void PS2_JoystickQuit(void)
{
    deinit_joystick_driver(true);
}

static bool PS2_GetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return false;
}

SDL_JoystickDriver SDL_PS2_JoystickDriver = {
    PS2_JoystickInit,
    PS2_JoystickGetCount,
    PS2_JoystickDetect,
    PS2_JoystickIsDevicePresent,
    PS2_JoystickGetDeviceName,
    PS2_JoystickGetDevicePath,
    PS2_JoystickGetDeviceSteamVirtualGamepadSlot,
    PS2_JoystickGetDevicePlayerIndex,
    PS2_JoystickSetDevicePlayerIndex,
    PS2_JoystickGetDeviceGUID,
    PS2_JoystickGetDeviceInstanceID,
    PS2_JoystickOpen,
    PS2_JoystickRumble,
    PS2_JoystickRumbleTriggers,
    PS2_JoystickSetLED,
    PS2_JoystickSendEffect,
    PS2_JoystickSetSensorsEnabled,
    PS2_JoystickUpdate,
    PS2_JoystickClose,
    PS2_JoystickQuit,
    PS2_GetGamepadMapping,
};

#endif // SDL_JOYSTICK_PS2
