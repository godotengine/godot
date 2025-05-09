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

#ifdef SDL_JOYSTICK_N3DS

// This is the N3DS implementation of the SDL joystick API

#include <3ds.h>

#include "../SDL_sysjoystick.h"

#define NB_BUTTONS 23

/*
  N3DS sticks values are roughly within +/-160
  which is too small to pass the jitter tolerance.
  This correction is applied to axis values
  so they fit better in SDL's value range.
*/
static inline int Correct_Axis_X(int X) {
    if (X > 160) {
        return SDL_JOYSTICK_AXIS_MAX;
    }
    else if (X < -160) {
        return -SDL_JOYSTICK_AXIS_MAX;
    }
    return (X * SDL_JOYSTICK_AXIS_MAX) / 160;
}

/*
  The Y axis needs to be flipped because SDL's "up"
  is reversed compared to libctru's "up"
*/
static inline int Correct_Axis_Y(int Y) {
    return Correct_Axis_X(-Y);
}

static void UpdateN3DSPressedButtons(Uint64 timestamp, SDL_Joystick *joystick);
static void UpdateN3DSReleasedButtons(Uint64 timestamp, SDL_Joystick *joystick);
static void UpdateN3DSCircle(Uint64 timestamp, SDL_Joystick *joystick);
static void UpdateN3DSCStick(Uint64 timestamp, SDL_Joystick *joystick);

static bool N3DS_JoystickInit(void)
{
    hidInit();
    SDL_PrivateJoystickAdded(1);
    return true;
}

static const char *N3DS_JoystickGetDeviceName(int device_index)
{
    return "Nintendo 3DS";
}

static int N3DS_JoystickGetCount(void)
{
    return 1;
}

static SDL_GUID N3DS_JoystickGetDeviceGUID(int device_index)
{
    SDL_GUID guid = SDL_CreateJoystickGUIDForName("Nintendo 3DS");
    return guid;
}

static SDL_JoystickID N3DS_JoystickGetDeviceInstanceID(int device_index)
{
    return device_index + 1;
}

static bool N3DS_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    joystick->nbuttons = NB_BUTTONS;
    joystick->naxes = 4;
    joystick->nhats = 0;

    return true;
}

static bool N3DS_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static void N3DS_JoystickUpdate(SDL_Joystick *joystick)
{
    Uint64 timestamp = SDL_GetTicksNS();

    UpdateN3DSPressedButtons(timestamp, joystick);
    UpdateN3DSReleasedButtons(timestamp, joystick);
    UpdateN3DSCircle(timestamp, joystick);
    UpdateN3DSCStick(timestamp, joystick);
}

static void UpdateN3DSPressedButtons(Uint64 timestamp, SDL_Joystick *joystick)
{
    static u32 previous_state = 0;
    u32 updated_down;
    u32 current_state = hidKeysDown();
    updated_down = previous_state ^ current_state;
    if (updated_down) {
        for (Uint8 i = 0; i < joystick->nbuttons; i++) {
            if (current_state & BIT(i) & updated_down) {
                SDL_SendJoystickButton(timestamp, joystick, i, true);
            }
        }
    }
    previous_state = current_state;
}

static void UpdateN3DSReleasedButtons(Uint64 timestamp, SDL_Joystick *joystick)
{
    static u32 previous_state = 0;
    u32 updated_up;
    u32 current_state = hidKeysUp();
    updated_up = previous_state ^ current_state;
    if (updated_up) {
        for (Uint8 i = 0; i < joystick->nbuttons; i++) {
            if (current_state & BIT(i) & updated_up) {
                SDL_SendJoystickButton(timestamp, joystick, i, false);
            }
        }
    }
    previous_state = current_state;
}

static void UpdateN3DSCircle(Uint64 timestamp, SDL_Joystick *joystick)
{
    static circlePosition previous_state = { 0, 0 };
    circlePosition current_state;
    hidCircleRead(&current_state);
    if (previous_state.dx != current_state.dx) {
        SDL_SendJoystickAxis(timestamp, joystick,
                                0,
                                Correct_Axis_X(current_state.dx));
    }
    if (previous_state.dy != current_state.dy) {
        SDL_SendJoystickAxis(timestamp, joystick,
                                1,
                                Correct_Axis_Y(current_state.dy));
    }
    previous_state = current_state;
}

static void UpdateN3DSCStick(Uint64 timestamp, SDL_Joystick *joystick)
{
    static circlePosition previous_state = { 0, 0 };
    circlePosition current_state;
    hidCstickRead(&current_state);
    if (previous_state.dx != current_state.dx) {
        SDL_SendJoystickAxis(timestamp, joystick,
                                2,
                                Correct_Axis_X(current_state.dx));
    }
    if (previous_state.dy != current_state.dy) {
        SDL_SendJoystickAxis(timestamp, joystick,
                                3,
                                Correct_Axis_Y(current_state.dy));
    }
    previous_state = current_state;
}

static void N3DS_JoystickClose(SDL_Joystick *joystick)
{
}

static void N3DS_JoystickQuit(void)
{
    hidExit();
}

static bool N3DS_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    // There is only one possible mapping.
    *out = (SDL_GamepadMapping){
        .a = { EMappingKind_Button, 0 },
        .b = { EMappingKind_Button, 1 },
        .x = { EMappingKind_Button, 10 },
        .y = { EMappingKind_Button, 11 },
        .back = { EMappingKind_Button, 2 },
        .guide = { EMappingKind_None, 255 },
        .start = { EMappingKind_Button, 3 },
        .leftstick = { EMappingKind_None, 255 },
        .rightstick = { EMappingKind_None, 255 },
        .leftshoulder = { EMappingKind_Button, 9 },
        .rightshoulder = { EMappingKind_Button, 8 },
        .dpup = { EMappingKind_Button, 6 },
        .dpdown = { EMappingKind_Button, 7 },
        .dpleft = { EMappingKind_Button, 5 },
        .dpright = { EMappingKind_Button, 4 },
        .misc1 = { EMappingKind_None, 255 },
        .right_paddle1 = { EMappingKind_None, 255 },
        .left_paddle1 = { EMappingKind_None, 255 },
        .right_paddle2 = { EMappingKind_None, 255 },
        .left_paddle2 = { EMappingKind_None, 255 },
        .leftx = { EMappingKind_Axis, 0 },
        .lefty = { EMappingKind_Axis, 1 },
        .rightx = { EMappingKind_Axis, 2 },
        .righty = { EMappingKind_Axis, 3 },
        .lefttrigger = { EMappingKind_Button, 14 },
        .righttrigger = { EMappingKind_Button, 15 },
    };
    return true;
}

static void N3DS_JoystickDetect(void)
{
}

static bool N3DS_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    // We don't override any other drivers
    return false;
}

static const char *N3DS_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int N3DS_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int N3DS_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void N3DS_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static bool N3DS_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    return SDL_Unsupported();
}

static bool N3DS_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static bool N3DS_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool N3DS_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

SDL_JoystickDriver SDL_N3DS_JoystickDriver = {
    N3DS_JoystickInit,
    N3DS_JoystickGetCount,
    N3DS_JoystickDetect,
    N3DS_JoystickIsDevicePresent,
    N3DS_JoystickGetDeviceName,
    N3DS_JoystickGetDevicePath,
    N3DS_JoystickGetDeviceSteamVirtualGamepadSlot,
    N3DS_JoystickGetDevicePlayerIndex,
    N3DS_JoystickSetDevicePlayerIndex,
    N3DS_JoystickGetDeviceGUID,
    N3DS_JoystickGetDeviceInstanceID,
    N3DS_JoystickOpen,
    N3DS_JoystickRumble,
    N3DS_JoystickRumbleTriggers,
    N3DS_JoystickSetLED,
    N3DS_JoystickSendEffect,
    N3DS_JoystickSetSensorsEnabled,
    N3DS_JoystickUpdate,
    N3DS_JoystickClose,
    N3DS_JoystickQuit,
    N3DS_JoystickGetGamepadMapping
};

#endif // SDL_JOYSTICK_N3DS
