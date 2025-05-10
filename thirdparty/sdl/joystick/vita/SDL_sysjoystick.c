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

#ifdef SDL_JOYSTICK_VITA

// This is the PSVita implementation of the SDL joystick API
#include <psp2/types.h>
#include <psp2/ctrl.h>
#include <psp2/kernel/threadmgr.h>

#include <stdio.h> // For the definition of NULL
#include <stdlib.h>

#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"

// Current pad state
static SceCtrlData pad0 = { .lx = 0, .ly = 0, .rx = 0, .ry = 0, .lt = 0, .rt = 0, .buttons = 0 };
static SceCtrlData pad1 = { .lx = 0, .ly = 0, .rx = 0, .ry = 0, .lt = 0, .rt = 0, .buttons = 0 };
static SceCtrlData pad2 = { .lx = 0, .ly = 0, .rx = 0, .ry = 0, .lt = 0, .rt = 0, .buttons = 0 };
static SceCtrlData pad3 = { .lx = 0, .ly = 0, .rx = 0, .ry = 0, .lt = 0, .rt = 0, .buttons = 0 };

static int ext_port_map[4] = { 1, 2, 3, 4 }; // index: SDL joy number, entry: Vita port number. For external controllers

static int SDL_numjoysticks = 1;

static const unsigned int ext_button_map[] = {
    SCE_CTRL_TRIANGLE,
    SCE_CTRL_CIRCLE,
    SCE_CTRL_CROSS,
    SCE_CTRL_SQUARE,
    SCE_CTRL_L1,
    SCE_CTRL_R1,
    SCE_CTRL_DOWN,
    SCE_CTRL_LEFT,
    SCE_CTRL_UP,
    SCE_CTRL_RIGHT,
    SCE_CTRL_SELECT,
    SCE_CTRL_START,
    SCE_CTRL_L2,
    SCE_CTRL_R2,
    SCE_CTRL_L3,
    SCE_CTRL_R3
};

static int analog_map[256]; // Map analog inputs to -32768 -> 32767

// 4 points define the bezier-curve.
// The Vita has a good amount of analog travel, so use a linear curve
static SDL_Point a = { 0, 0 };
static SDL_Point b = { 0, 0 };
static SDL_Point c = { 128, 32767 };
static SDL_Point d = { 128, 32767 };

// simple linear interpolation between two points
static SDL_INLINE void lerp(SDL_Point *dest, const SDL_Point *first, const SDL_Point *second, float t)
{
    dest->x = first->x + (int)((second->x - first->x) * t);
    dest->y = first->y + (int)((second->y - first->y) * t);
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
static bool VITA_JoystickInit(void)
{
    int i;
    SceCtrlPortInfo myPortInfo;

    // Setup input
    sceCtrlSetSamplingMode(SCE_CTRL_MODE_ANALOG_WIDE);
    sceCtrlSetSamplingModeExt(SCE_CTRL_MODE_ANALOG_WIDE);

    /* Create an accurate map from analog inputs (0 to 255)
       to SDL joystick positions (-32768 to 32767) */
    for (i = 0; i < 128; i++) {
        float t = (float)i / 127.0f;
        analog_map[i + 128] = calc_bezier_y(t);
        analog_map[127 - i] = -1 * analog_map[i + 128];
    }

    // Assume we have at least one controller, even when nothing is paired
    // This way the user can jump in, pair a controller
    // and control things immediately even if it is paired
    // after the app has already started.

    SDL_numjoysticks = 1;
    SDL_PrivateJoystickAdded(SDL_numjoysticks);

    // How many additional paired controllers are there?
    sceCtrlGetControllerPortInfo(&myPortInfo);

    // On Vita TV, port 0 and 1 are the same controller
    // and that is the first one, so start at port 2
    for (i = 2; i <= 4; i++) {
        if (myPortInfo.port[i] != SCE_CTRL_TYPE_UNPAIRED) {
            ++SDL_numjoysticks;
            SDL_PrivateJoystickAdded(SDL_numjoysticks);
        }
    }
    return SDL_numjoysticks;
}

static int VITA_JoystickGetCount(void)
{
    return SDL_numjoysticks;
}

static void VITA_JoystickDetect(void)
{
}

static bool VITA_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    // We don't override any other drivers
    return false;
}

// Function to perform the mapping from device index to the instance id for this index
static SDL_JoystickID VITA_JoystickGetDeviceInstanceID(int device_index)
{
    return device_index + 1;
}

static const char *VITA_JoystickGetDeviceName(int index)
{
    if (index == 0) {
        return "PSVita Controller";
    }

    if (index == 1) {
        return "PSVita Controller";
    }

    if (index == 2) {
        return "PSVita Controller";
    }

    if (index == 3) {
        return "PSVita Controller";
    }

    SDL_SetError("No joystick available with that index");
    return NULL;
}

static const char *VITA_JoystickGetDevicePath(int index)
{
    return NULL;
}

static int VITA_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int VITA_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void VITA_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

/* Function to open a joystick for use.
   The joystick to open is specified by the device index.
   This should fill the nbuttons and naxes fields of the joystick structure.
   It returns 0, or -1 if there is an error.
 */
static bool VITA_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    joystick->nbuttons = SDL_arraysize(ext_button_map);
    joystick->naxes = 6;
    joystick->nhats = 0;

    SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_RGB_LED_BOOLEAN, true);
    SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_RUMBLE_BOOLEAN, true);

    return true;
}

/* Function to update the state of a joystick - called as a device poll.
 * This function shouldn't update the joystick structure directly,
 * but instead should call SDL_PrivateJoystick*() to deliver events
 * and update joystick device state.
 */
static void VITA_JoystickUpdate(SDL_Joystick *joystick)
{
    int i;
    unsigned int buttons;
    unsigned int changed;
    unsigned char lx, ly, rx, ry, lt, rt;
    static unsigned int old_buttons[] = { 0, 0, 0, 0 };
    static unsigned char old_lx[] = { 0, 0, 0, 0 };
    static unsigned char old_ly[] = { 0, 0, 0, 0 };
    static unsigned char old_rx[] = { 0, 0, 0, 0 };
    static unsigned char old_ry[] = { 0, 0, 0, 0 };
    static unsigned char old_lt[] = { 0, 0, 0, 0 };
    static unsigned char old_rt[] = { 0, 0, 0, 0 };
    SceCtrlData *pad = NULL;
    Uint64 timestamp = SDL_GetTicksNS();

    int index = (int)SDL_GetJoystickID(joystick) - 1;

    if (index == 0)
        pad = &pad0;
    else if (index == 1)
        pad = &pad1;
    else if (index == 2)
        pad = &pad2;
    else if (index == 3)
        pad = &pad3;
    else
        return;

    if (index == 0) {
        if (sceCtrlPeekBufferPositive2(ext_port_map[index], pad, 1) < 0) {
            // on vita fallback to port 0
            sceCtrlPeekBufferPositive2(0, pad, 1);
        }
    } else {
        sceCtrlPeekBufferPositive2(ext_port_map[index], pad, 1);
    }

    buttons = pad->buttons;

    lx = pad->lx;
    ly = pad->ly;
    rx = pad->rx;
    ry = pad->ry;
    lt = pad->lt;
    rt = pad->rt;

    // Axes

    if (old_lx[index] != lx) {
        SDL_SendJoystickAxis(timestamp, joystick, 0, analog_map[lx]);
        old_lx[index] = lx;
    }
    if (old_ly[index] != ly) {
        SDL_SendJoystickAxis(timestamp, joystick, 1, analog_map[ly]);
        old_ly[index] = ly;
    }
    if (old_rx[index] != rx) {
        SDL_SendJoystickAxis(timestamp, joystick, 2, analog_map[rx]);
        old_rx[index] = rx;
    }
    if (old_ry[index] != ry) {
        SDL_SendJoystickAxis(timestamp, joystick, 3, analog_map[ry]);
        old_ry[index] = ry;
    }

    if (old_lt[index] != lt) {
        SDL_SendJoystickAxis(timestamp, joystick, 4, analog_map[lt]);
        old_lt[index] = lt;
    }
    if (old_rt[index] != rt) {
        SDL_SendJoystickAxis(timestamp, joystick, 5, analog_map[rt]);
        old_rt[index] = rt;
    }

    // Buttons
    changed = old_buttons[index] ^ buttons;
    old_buttons[index] = buttons;

    if (changed) {
        for (i = 0; i < SDL_arraysize(ext_button_map); i++) {
            if (changed & ext_button_map[i]) {
                bool down = ((buttons & ext_button_map[i]) != 0);
                SDL_SendJoystickButton(timestamp, joystick, i, down);
            }
        }
    }
}

// Function to close a joystick after use
static void VITA_JoystickClose(SDL_Joystick *joystick)
{
}

// Function to perform any system-specific joystick related cleanup
static void VITA_JoystickQuit(void)
{
}

static SDL_GUID VITA_JoystickGetDeviceGUID(int device_index)
{
    // the GUID is just the name for now
    const char *name = VITA_JoystickGetDeviceName(device_index);
    return SDL_CreateJoystickGUIDForName(name);
}

static bool VITA_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    int index = (int)SDL_GetJoystickID(joystick) - 1;
    SceCtrlActuator act;

    if (index < 0 || index > 3) {
        return false;
    }
    SDL_zero(act);
    act.small = high_frequency_rumble / 256;
    act.large = low_frequency_rumble / 256;
    if (sceCtrlSetActuator(ext_port_map[index], &act) < 0) {
        return SDL_Unsupported();
    }
    return true;
}

static bool VITA_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left, Uint16 right)
{
    return SDL_Unsupported();
}

static bool VITA_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    int index = (int)SDL_GetJoystickID(joystick) - 1;
    if (index < 0 || index > 3) {
        return false;
    }
    if (sceCtrlSetLightBar(ext_port_map[index], red, green, blue) < 0) {
        return SDL_Unsupported();
    }
    return true;
}

static bool VITA_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool VITA_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static bool VITA_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return false;
}

SDL_JoystickDriver SDL_VITA_JoystickDriver = {
    VITA_JoystickInit,
    VITA_JoystickGetCount,
    VITA_JoystickDetect,
    VITA_JoystickIsDevicePresent,
    VITA_JoystickGetDeviceName,
    VITA_JoystickGetDevicePath,
    VITA_JoystickGetDeviceSteamVirtualGamepadSlot,
    VITA_JoystickGetDevicePlayerIndex,
    VITA_JoystickSetDevicePlayerIndex,
    VITA_JoystickGetDeviceGUID,
    VITA_JoystickGetDeviceInstanceID,
    VITA_JoystickOpen,
    VITA_JoystickRumble,
    VITA_JoystickRumbleTriggers,
    VITA_JoystickSetLED,
    VITA_JoystickSendEffect,
    VITA_JoystickSetSensorsEnabled,
    VITA_JoystickUpdate,
    VITA_JoystickClose,
    VITA_JoystickQuit,
    VITA_JoystickGetGamepadMapping,
};

#endif // SDL_JOYSTICK_VITA
