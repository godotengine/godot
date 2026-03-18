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

#ifndef SDL_JOYSTICK_IOS_H
#define SDL_JOYSTICK_IOS_H

#include "../SDL_sysjoystick.h"

#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>

@class GCController;

typedef struct joystick_hwdata
{
    GCController __unsafe_unretained *controller;
    void *rumble;
    int pause_button_index;
    Uint64 pause_button_pressed;

    char *name;
    SDL_Joystick *joystick;
    SDL_JoystickID instance_id;
    SDL_GUID guid;

    int naxes;
    int nbuttons;
    int nhats;
    Uint32 button_mask;
    bool is_xbox;
    bool is_ps4;
    bool is_ps5;
    bool is_switch_pro;
    bool is_switch_joycon_pair;
    bool is_switch_joyconL;
    bool is_switch_joyconR;
    bool is_stadia;
    bool is_backbone_one;
    int is_siri_remote;

    NSArray __unsafe_unretained *axes;
    NSArray __unsafe_unretained *buttons;

    bool has_dualshock_touchpad;
    bool has_xbox_paddles;
    bool has_xbox_share_button;
    bool has_nintendo_buttons;

    struct joystick_hwdata *next;
} joystick_hwdata;

typedef joystick_hwdata SDL_JoystickDeviceItem;

#endif // SDL_JOYSTICK_IOS_H
