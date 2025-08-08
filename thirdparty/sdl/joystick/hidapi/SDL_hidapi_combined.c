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
// This driver supports the Nintendo Switch Joy-Cons pair controllers
#include "SDL_internal.h"

#ifdef SDL_JOYSTICK_HIDAPI

#include "SDL_hidapijoystick_c.h"
#include "../SDL_sysjoystick.h"

static void HIDAPI_DriverCombined_RegisterHints(SDL_HintCallback callback, void *userdata)
{
}

static void HIDAPI_DriverCombined_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
}

static bool HIDAPI_DriverCombined_IsEnabled(void)
{
    return true;
}

static bool HIDAPI_DriverCombined_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    // This is always explicitly created for combined devices
    return false;
}

static bool HIDAPI_DriverCombined_InitDevice(SDL_HIDAPI_Device *device)
{
    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverCombined_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverCombined_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static bool HIDAPI_DriverCombined_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    int i;
    char *serial = NULL, *new_serial;
    size_t serial_length = 0, new_length;

    SDL_AssertJoysticksLocked();

    for (i = 0; i < device->num_children; ++i) {
        SDL_HIDAPI_Device *child = device->children[i];
        if (!child->driver->OpenJoystick(child, joystick)) {
            child->broken = true;

            while (i-- > 0) {
                child = device->children[i];
                child->driver->CloseJoystick(child, joystick);
            }
            if (serial) {
                SDL_free(serial);
            }
            return false;
        }

        // Extend the serial number with the child serial number
        if (joystick->serial) {
            new_length = serial_length + 1 + SDL_strlen(joystick->serial);
            new_serial = (char *)SDL_realloc(serial, new_length);
            if (new_serial) {
                if (serial) {
                    SDL_strlcat(new_serial, ",", new_length);
                    SDL_strlcat(new_serial, joystick->serial, new_length);
                } else {
                    SDL_strlcpy(new_serial, joystick->serial, new_length);
                }
                serial = new_serial;
                serial_length = new_length;
            }
            SDL_free(joystick->serial);
            joystick->serial = NULL;
        }
    }

    // Update the joystick with the combined serial numbers
    if (joystick->serial) {
        SDL_free(joystick->serial);
    }
    joystick->serial = serial;

    return true;
}

static bool HIDAPI_DriverCombined_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    int i;
    bool result = false;

    for (i = 0; i < device->num_children; ++i) {
        SDL_HIDAPI_Device *child = device->children[i];
        if (child->driver->RumbleJoystick(child, joystick, low_frequency_rumble, high_frequency_rumble)) {
            result = true;
        }
    }
    return result;
}

static bool HIDAPI_DriverCombined_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    int i;
    bool result = false;

    for (i = 0; i < device->num_children; ++i) {
        SDL_HIDAPI_Device *child = device->children[i];
        if (child->driver->RumbleJoystickTriggers(child, joystick, left_rumble, right_rumble)) {
            result = true;
        }
    }
    return result;
}

static Uint32 HIDAPI_DriverCombined_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    int i;
    Uint32 caps = 0;

    for (i = 0; i < device->num_children; ++i) {
        SDL_HIDAPI_Device *child = device->children[i];
        caps |= child->driver->GetJoystickCapabilities(child, joystick);
    }
    return caps;
}

static bool HIDAPI_DriverCombined_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    int i;
    bool result = false;

    for (i = 0; i < device->num_children; ++i) {
        SDL_HIDAPI_Device *child = device->children[i];
        if (child->driver->SetJoystickLED(child, joystick, red, green, blue)) {
            result = true;
        }
    }
    return result;
}

static bool HIDAPI_DriverCombined_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverCombined_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    int i;
    bool result = false;

    for (i = 0; i < device->num_children; ++i) {
        SDL_HIDAPI_Device *child = device->children[i];
        if (child->driver->SetJoystickSensorsEnabled(child, joystick, enabled)) {
            result = true;
        }
    }
    return result;
}

static bool HIDAPI_DriverCombined_UpdateDevice(SDL_HIDAPI_Device *device)
{
    int i;
    int result = true;

    for (i = 0; i < device->num_children; ++i) {
        SDL_HIDAPI_Device *child = device->children[i];
        if (!child->driver->UpdateDevice(child)) {
            result = false;
        }
    }
    return result;
}

static void HIDAPI_DriverCombined_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    int i;

    for (i = 0; i < device->num_children; ++i) {
        SDL_HIDAPI_Device *child = device->children[i];
        child->driver->CloseJoystick(child, joystick);
    }
}

static void HIDAPI_DriverCombined_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverCombined = {
    "SDL_JOYSTICK_HIDAPI_COMBINED",
    true,
    HIDAPI_DriverCombined_RegisterHints,
    HIDAPI_DriverCombined_UnregisterHints,
    HIDAPI_DriverCombined_IsEnabled,
    HIDAPI_DriverCombined_IsSupportedDevice,
    HIDAPI_DriverCombined_InitDevice,
    HIDAPI_DriverCombined_GetDevicePlayerIndex,
    HIDAPI_DriverCombined_SetDevicePlayerIndex,
    HIDAPI_DriverCombined_UpdateDevice,
    HIDAPI_DriverCombined_OpenJoystick,
    HIDAPI_DriverCombined_RumbleJoystick,
    HIDAPI_DriverCombined_RumbleJoystickTriggers,
    HIDAPI_DriverCombined_GetJoystickCapabilities,
    HIDAPI_DriverCombined_SetJoystickLED,
    HIDAPI_DriverCombined_SendJoystickEffect,
    HIDAPI_DriverCombined_SetJoystickSensorsEnabled,
    HIDAPI_DriverCombined_CloseJoystick,
    HIDAPI_DriverCombined_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI
