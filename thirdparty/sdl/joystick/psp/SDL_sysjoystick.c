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

#ifdef SDL_JOYSTICK_PSP

// This is the PSP implementation of the SDL joystick API
#include <pspctrl.h>

#include <stdio.h> // For the definition of NULL
#include <stdlib.h>

#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"

// Current pad state
static SceCtrlData pad = { .Lx = 0, .Ly = 0, .Buttons = 0 };
static const enum PspCtrlButtons button_map[] = {
    PSP_CTRL_TRIANGLE, PSP_CTRL_CIRCLE, PSP_CTRL_CROSS, PSP_CTRL_SQUARE,
    PSP_CTRL_LTRIGGER, PSP_CTRL_RTRIGGER,
    PSP_CTRL_DOWN, PSP_CTRL_LEFT, PSP_CTRL_UP, PSP_CTRL_RIGHT,
    PSP_CTRL_SELECT, PSP_CTRL_START, PSP_CTRL_HOME, PSP_CTRL_HOLD
};
static int analog_map[256]; // Map analog inputs to -32768 -> 32767

// 4 points define the bezier-curve.
static SDL_Point a = { 0, 0 };
static SDL_Point b = { 50, 0 };
static SDL_Point c = { 78, 32767 };
static SDL_Point d = { 128, 32767 };

// simple linear interpolation between two points
static SDL_INLINE void lerp(SDL_Point *dest, const SDL_Point *pt_a, const SDL_Point *pt_b, float t)
{
    dest->x = pt_a->x + (int)((pt_b->x - pt_a->x) * t);
    dest->y = pt_a->y + (int)((pt_b->y - pt_a->y) * t);
}

// evaluate a point on a bezier-curve. t goes from 0 to 1.0
static int calc_bezier_y(float t)
{
    SDL_Point ab, bc, cd, abbc, bccd, dest;
    lerp(&ab, &a, &b, t);         // point between a and b
    lerp(&bc, &b, &c, t);         // point between b and c
    lerp(&cd, &c, &d, t);         // point between c and d
    lerp(&abbc, &ab, &bc, t);     // point between ab and bc
    lerp(&bccd, &bc, &cd, t);     // point between bc and cd
    lerp(&dest, &abbc, &bccd, t); // point on the bezier-curve
    return dest.y;
}

/* Function to scan the system for joysticks.
 * Joystick 0 should be the system default joystick.
 * It should return number of joysticks, or -1 on an unrecoverable fatal error.
 */
static bool PSP_JoystickInit(void)
{
    int i;

    // Setup input
    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_ANALOG);

    /* Create an accurate map from analog inputs (0 to 255)
       to SDL joystick positions (-32768 to 32767) */
    for (i = 0; i < 128; i++) {
        float t = (float)i / 127.0f;
        analog_map[i + 128] = calc_bezier_y(t);
        analog_map[127 - i] = -1 * analog_map[i + 128];
    }

    SDL_PrivateJoystickAdded(1);

    return 1;
}

static int PSP_JoystickGetCount(void)
{
    return 1;
}

static void PSP_JoystickDetect(void)
{
}

static bool PSP_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    // We don't override any other drivers
    return false;
}

// Function to get the device-dependent name of a joystick
static const char *PSP_JoystickGetDeviceName(int device_index)
{
    if (device_index == 0) {
        return "PSP builtin joypad";
    }

    SDL_SetError("No joystick available with that index");
    return NULL;
}

static const char *PSP_JoystickGetDevicePath(int index)
{
    return NULL;
}

static int PSP_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int PSP_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void PSP_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static SDL_GUID PSP_JoystickGetDeviceGUID(int device_index)
{
    // the GUID is just the name for now
    const char *name = PSP_JoystickGetDeviceName(device_index);
    return SDL_CreateJoystickGUIDForName(name);
}

// Function to perform the mapping from device index to the instance id for this index
static SDL_JoystickID PSP_JoystickGetDeviceInstanceID(int device_index)
{
    return device_index + 1;
}

/* Function to open a joystick for use.
   The joystick to open is specified by the device index.
   This should fill the nbuttons and naxes fields of the joystick structure.
   It returns 0, or -1 if there is an error.
 */
static bool PSP_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    joystick->nbuttons = SDL_arraysize(button_map);
    joystick->naxes = 2;
    joystick->nhats = 0;

    return true;
}

static bool PSP_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    return SDL_Unsupported();
}

static bool PSP_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static bool PSP_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool PSP_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool PSP_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

/* Function to update the state of a joystick - called as a device poll.
 * This function shouldn't update the joystick structure directly,
 * but instead should call SDL_PrivateJoystick*() to deliver events
 * and update joystick device state.
 */
static void PSP_JoystickUpdate(SDL_Joystick *joystick)
{
    int i;
    enum PspCtrlButtons buttons;
    enum PspCtrlButtons changed;
    unsigned char x, y;
    static enum PspCtrlButtons old_buttons = 0;
    static unsigned char old_x = 0, old_y = 0;
    Uint64 timestamp = SDL_GetTicksNS();

    if (sceCtrlPeekBufferPositive(&pad, 1) <= 0) {
        return;
    }
    buttons = pad.Buttons;
    x = pad.Lx;
    y = pad.Ly;

    // Axes
    if (old_x != x) {
        SDL_SendJoystickAxis(timestamp, joystick, 0, analog_map[x]);
        old_x = x;
    }
    if (old_y != y) {
        SDL_SendJoystickAxis(timestamp, joystick, 1, analog_map[y]);
        old_y = y;
    }

    // Buttons
    changed = old_buttons ^ buttons;
    old_buttons = buttons;
    if (changed) {
        for (i = 0; i < SDL_arraysize(button_map); i++) {
            if (changed & button_map[i]) {
                bool down = ((buttons & button_map[i]) != 0);
                SDL_SendJoystickButton(timestamp,
                    joystick, i, down);
            }
        }
    }
}

// Function to close a joystick after use
static void PSP_JoystickClose(SDL_Joystick *joystick)
{
}

// Function to perform any system-specific joystick related cleanup
static void PSP_JoystickQuit(void)
{
}

static bool PSP_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return false;
}

SDL_JoystickDriver SDL_PSP_JoystickDriver = {
    PSP_JoystickInit,
    PSP_JoystickGetCount,
    PSP_JoystickDetect,
    PSP_JoystickIsDevicePresent,
    PSP_JoystickGetDeviceName,
    PSP_JoystickGetDevicePath,
    PSP_JoystickGetDeviceSteamVirtualGamepadSlot,
    PSP_JoystickGetDevicePlayerIndex,
    PSP_JoystickSetDevicePlayerIndex,
    PSP_JoystickGetDeviceGUID,
    PSP_JoystickGetDeviceInstanceID,
    PSP_JoystickOpen,
    PSP_JoystickRumble,
    PSP_JoystickRumbleTriggers,
    PSP_JoystickSetLED,
    PSP_JoystickSendEffect,
    PSP_JoystickSetSensorsEnabled,
    PSP_JoystickUpdate,
    PSP_JoystickClose,
    PSP_JoystickQuit,
    PSP_JoystickGetGamepadMapping
};

#endif // SDL_JOYSTICK_PSP
