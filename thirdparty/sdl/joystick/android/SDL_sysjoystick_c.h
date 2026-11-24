/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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

#ifdef SDL_JOYSTICK_ANDROID

#ifndef SDL_sysjoystick_c_h_
#define SDL_sysjoystick_c_h_

#include "../SDL_sysjoystick.h"

extern bool Android_OnPadDown(int device_id, int keycode, int scancode);
extern bool Android_OnPadUp(int device_id, int keycode, int scancode);
extern bool Android_OnJoy(int device_id, int axisnum, float value);
extern bool Android_OnHat(int device_id, int hat_id, int x, int y);
extern void Android_OnJoySensor(int device_id, int sensor_type, Uint64 sensor_timestamp, float x, float y, float z);
extern void Android_AddJoystick(int device_id, const char *name, const char *desc, int vendor_id, int product_id, int button_mask, int naxes, int axis_mask, int nhats,
   bool can_rumble, bool has_rgb_led, bool has_accelerometer, bool has_gyroscope);
extern void Android_RemoveJoystick(int device_id);

// A linked list of available joysticks
typedef struct SDL_joylist_item
{
    int device_instance;
    int device_id; // Android's device id
    char *name;    // "SideWinder 3D Pro" or whatever
    SDL_GUID guid;
    SDL_Joystick *joystick;
    int nbuttons, naxes, nhats;
    int dpad_state;
    bool can_rumble;
    bool has_rgb_led;
    bool has_accelerometer;
    bool has_gyroscope;

    struct SDL_joylist_item *next;
} SDL_joylist_item;

typedef SDL_joylist_item joystick_hwdata;

#endif // SDL_sysjoystick_c_h_

#endif // SDL_JOYSTICK_ANDROID
