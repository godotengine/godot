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

#ifdef SDL_JOYSTICK_HAIKU

// This is the Haiku implementation of the SDL joystick API

#include <support/String.h>
#include <device/Joystick.h>

extern "C"
{

#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"


// The maximum number of joysticks we'll detect
#define MAX_JOYSTICKS   16

// A list of available joysticks
    static char *SDL_joyport[MAX_JOYSTICKS];
    static char *SDL_joyname[MAX_JOYSTICKS];

// The private structure used to keep track of a joystick
    struct joystick_hwdata
    {
        BJoystick *stick;
        uint8 *new_hats;
        int16 *new_axes;
    };

    static int numjoysticks = 0;

    static bool HAIKU_JoystickInit(void)
    {
        BJoystick joystick;
        int i;
        int32 nports;
        char name[B_OS_NAME_LENGTH];

        // Search for attached joysticks
        nports = joystick.CountDevices();
        numjoysticks = 0;
        SDL_memset(SDL_joyport, 0, sizeof(SDL_joyport));
        SDL_memset(SDL_joyname, 0, sizeof(SDL_joyname));
        for (i = 0; (numjoysticks < MAX_JOYSTICKS) && (i < nports); ++i) {
            if (joystick.GetDeviceName(i, name) == B_OK) {
                if (joystick.Open(name) != B_ERROR) {
                    BString stick_name;
                    joystick.GetControllerName(&stick_name);
                    SDL_joyport[numjoysticks] = SDL_strdup(name);
                    SDL_joyname[numjoysticks] = SDL_CreateJoystickName(0, 0, NULL, stick_name.String());
                    numjoysticks++;
                    joystick.Close();

                    SDL_PrivateJoystickAdded(numjoysticks);
                }
            }
        }
        return true;
    }

    static int HAIKU_JoystickGetCount(void)
    {
        return numjoysticks;
    }

    static void HAIKU_JoystickDetect(void)
    {
    }

    static bool HAIKU_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
    {
        // We don't override any other drivers
        return false;
    }

    static const char *HAIKU_JoystickGetDeviceName(int device_index)
    {
        return SDL_joyname[device_index];
    }

    static const char *HAIKU_JoystickGetDevicePath(int device_index)
    {
        return SDL_joyport[device_index];
    }

    static int HAIKU_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
    {
        return -1;
    }

    static int HAIKU_JoystickGetDevicePlayerIndex(int device_index)
    {
        return -1;
    }

    static void HAIKU_JoystickSetDevicePlayerIndex(int device_index, int player_index)
    {
    }

    static SDL_JoystickID HAIKU_JoystickGetDeviceInstanceID(int device_index)
    {
        return device_index + 1;
    }

    static void HAIKU_JoystickClose(SDL_Joystick *joystick);

    static bool HAIKU_JoystickOpen(SDL_Joystick *joystick, int device_index)
    {
        BJoystick *stick;

        // Create the joystick data structure
        joystick->hwdata = (struct joystick_hwdata *) SDL_calloc(1, sizeof(*joystick->hwdata));
        if (joystick->hwdata == NULL) {
            return false;
        }
        stick = new BJoystick;
        joystick->hwdata->stick = stick;

        // Open the requested joystick for use
        if (stick->Open(SDL_joyport[device_index]) == B_ERROR) {
            HAIKU_JoystickClose(joystick);
            return SDL_SetError("Unable to open joystick");
        }

        // Set the joystick to calibrated mode
        stick->EnableCalibration();

        // Get the number of buttons, hats, and axes on the joystick
        joystick->nbuttons = stick->CountButtons();
        joystick->naxes = stick->CountAxes();
        joystick->nhats = stick->CountHats();

        joystick->hwdata->new_axes = (int16 *) SDL_calloc(joystick->naxes, sizeof(int16));
        joystick->hwdata->new_hats = (uint8 *) SDL_calloc(joystick->nhats, sizeof(uint8));
        if (!joystick->hwdata->new_hats || !joystick->hwdata->new_axes) {
            HAIKU_JoystickClose(joystick);
            return false;
        }

        // We're done!
        return true;
    }

/* Function to update the state of a joystick - called as a device poll.
 * This function shouldn't update the joystick structure directly,
 * but instead should call SDL_PrivateJoystick*() to deliver events
 * and update joystick device state.
 */
    static void HAIKU_JoystickUpdate(SDL_Joystick *joystick)
    {
        static const Uint8 hat_map[9] = {
            SDL_HAT_CENTERED,
            SDL_HAT_UP,
            SDL_HAT_RIGHTUP,
            SDL_HAT_RIGHT,
            SDL_HAT_RIGHTDOWN,
            SDL_HAT_DOWN,
            SDL_HAT_LEFTDOWN,
            SDL_HAT_LEFT,
            SDL_HAT_LEFTUP
        };

        BJoystick *stick;
        int i;
        int16 *axes;
        uint8 *hats;
        uint32 buttons;
        Uint64 timestamp = SDL_GetTicksNS();

        // Set up data pointers
        stick = joystick->hwdata->stick;
        axes = joystick->hwdata->new_axes;
        hats = joystick->hwdata->new_hats;

        // Get the new joystick state
        stick->Update();
        stick->GetAxisValues(axes);
        stick->GetHatValues(hats);
        buttons = stick->ButtonValues();

        // Generate axis motion events
        for (i = 0; i < joystick->naxes; ++i) {
            SDL_SendJoystickAxis(timestamp, joystick, i, axes[i]);
        }

        // Generate hat change events
        for (i = 0; i < joystick->nhats; ++i) {
            SDL_SendJoystickHat(timestamp, joystick, i, hat_map[hats[i]]);
        }

        // Generate button events
        for (i = 0; i < joystick->nbuttons; ++i) {
            bool down = ((buttons & 0x01) != 0);
            SDL_SendJoystickButton(timestamp, joystick, i, down);
            buttons >>= 1;
        }
    }

// Function to close a joystick after use
    static void HAIKU_JoystickClose(SDL_Joystick *joystick)
    {
        if (joystick->hwdata) {
            joystick->hwdata->stick->Close();
            delete joystick->hwdata->stick;
            SDL_free(joystick->hwdata->new_hats);
            SDL_free(joystick->hwdata->new_axes);
            SDL_free(joystick->hwdata);
        }
    }

// Function to perform any system-specific joystick related cleanup
    static void HAIKU_JoystickQuit(void)
    {
        int i;

        for (i = 0; i < numjoysticks; ++i) {
            SDL_free(SDL_joyport[i]);
        }
        SDL_joyport[0] = NULL;

        for (i = 0; i < numjoysticks; ++i) {
            SDL_free(SDL_joyname[i]);
        }
        SDL_joyname[0] = NULL;
    }

    static SDL_GUID HAIKU_JoystickGetDeviceGUID(int device_index)
    {
        // the GUID is just the name for now
        const char *name = HAIKU_JoystickGetDeviceName(device_index);
        return SDL_CreateJoystickGUIDForName(name);
    }

    static bool HAIKU_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
    {
        return SDL_Unsupported();
    }


    static bool HAIKU_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
    {
        return SDL_Unsupported();
    }

    static bool HAIKU_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
    {
        return false;
    }

    static bool HAIKU_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
    {
        return SDL_Unsupported();
    }


    static bool HAIKU_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
    {
        return SDL_Unsupported();
    }

    static bool HAIKU_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
    {
        return SDL_Unsupported();
    }

    SDL_JoystickDriver SDL_HAIKU_JoystickDriver =
    {
        HAIKU_JoystickInit,
        HAIKU_JoystickGetCount,
        HAIKU_JoystickDetect,
        HAIKU_JoystickIsDevicePresent,
        HAIKU_JoystickGetDeviceName,
        HAIKU_JoystickGetDevicePath,
        HAIKU_JoystickGetDeviceSteamVirtualGamepadSlot,
        HAIKU_JoystickGetDevicePlayerIndex,
        HAIKU_JoystickSetDevicePlayerIndex,
        HAIKU_JoystickGetDeviceGUID,
        HAIKU_JoystickGetDeviceInstanceID,
        HAIKU_JoystickOpen,
        HAIKU_JoystickRumble,
        HAIKU_JoystickRumbleTriggers,
        HAIKU_JoystickSetLED,
        HAIKU_JoystickSendEffect,
        HAIKU_JoystickSetSensorsEnabled,
        HAIKU_JoystickUpdate,
        HAIKU_JoystickClose,
        HAIKU_JoystickQuit,
        HAIKU_JoystickGetGamepadMapping
    };

}                              // extern "C"

#endif // SDL_JOYSTICK_HAIKU
