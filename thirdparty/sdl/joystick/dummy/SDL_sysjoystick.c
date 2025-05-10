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

#if defined(SDL_JOYSTICK_DUMMY) || defined(SDL_JOYSTICK_DISABLED)

// This is the dummy implementation of the SDL joystick API

#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"

static bool DUMMY_JoystickInit(void)
{
    return true;
}

static int DUMMY_JoystickGetCount(void)
{
    return 0;
}

static void DUMMY_JoystickDetect(void)
{
}

static bool DUMMY_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    return false;
}

static const char *DUMMY_JoystickGetDeviceName(int device_index)
{
    return NULL;
}

static const char *DUMMY_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int DUMMY_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int DUMMY_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void DUMMY_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static SDL_GUID DUMMY_JoystickGetDeviceGUID(int device_index)
{
    SDL_GUID guid;
    SDL_zero(guid);
    return guid;
}

static SDL_JoystickID DUMMY_JoystickGetDeviceInstanceID(int device_index)
{
    return 0;
}

static bool DUMMY_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    return SDL_SetError("Logic error: No joysticks available");
}

static bool DUMMY_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    return SDL_Unsupported();
}

static bool DUMMY_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static bool DUMMY_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool DUMMY_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool DUMMY_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static void DUMMY_JoystickUpdate(SDL_Joystick *joystick)
{
}

static void DUMMY_JoystickClose(SDL_Joystick *joystick)
{
}

static void DUMMY_JoystickQuit(void)
{
}

static bool DUMMY_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return false;
}

SDL_JoystickDriver SDL_DUMMY_JoystickDriver = {
    DUMMY_JoystickInit,
    DUMMY_JoystickGetCount,
    DUMMY_JoystickDetect,
    DUMMY_JoystickIsDevicePresent,
    DUMMY_JoystickGetDeviceName,
    DUMMY_JoystickGetDevicePath,
    DUMMY_JoystickGetDeviceSteamVirtualGamepadSlot,
    DUMMY_JoystickGetDevicePlayerIndex,
    DUMMY_JoystickSetDevicePlayerIndex,
    DUMMY_JoystickGetDeviceGUID,
    DUMMY_JoystickGetDeviceInstanceID,
    DUMMY_JoystickOpen,
    DUMMY_JoystickRumble,
    DUMMY_JoystickRumbleTriggers,
    DUMMY_JoystickSetLED,
    DUMMY_JoystickSendEffect,
    DUMMY_JoystickSetSensorsEnabled,
    DUMMY_JoystickUpdate,
    DUMMY_JoystickClose,
    DUMMY_JoystickQuit,
    DUMMY_JoystickGetGamepadMapping
};

#endif // SDL_JOYSTICK_DUMMY || SDL_JOYSTICK_DISABLED
