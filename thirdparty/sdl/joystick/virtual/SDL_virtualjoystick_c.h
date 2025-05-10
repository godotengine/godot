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

#ifndef SDL_VIRTUALJOYSTICK_C_H
#define SDL_VIRTUALJOYSTICK_C_H

#ifdef SDL_JOYSTICK_VIRTUAL

#include "../SDL_sysjoystick.h"

#define AXES_CHANGED        0x00000001
#define BALLS_CHANGED       0x00000002
#define BUTTONS_CHANGED     0x00000004
#define HATS_CHANGED        0x00000008
#define TOUCHPADS_CHANGED   0x00000010

/**
 * Data for a virtual, software-only joystick.
 */
typedef struct VirtualSensorEvent
{
    SDL_SensorType type;
    Uint64 sensor_timestamp;
    float data[3];
    int num_values;
} VirtualSensorEvent;

typedef struct joystick_hwdata
{
    SDL_JoystickID instance_id;
    bool attached;
    char *name;
    SDL_JoystickType type;
    SDL_GUID guid;
    SDL_VirtualJoystickDesc desc;
    Uint32 changes;
    Sint16 *axes;
    bool *buttons;
    Uint8 *hats;
    SDL_JoystickBallData *balls;
    SDL_JoystickTouchpadInfo *touchpads;
    SDL_JoystickSensorInfo *sensors;
    bool sensors_enabled;
    int num_sensor_events;
    int max_sensor_events;
    VirtualSensorEvent *sensor_events;

    SDL_Joystick *joystick;

    struct joystick_hwdata *next;
} joystick_hwdata;

extern SDL_JoystickID SDL_JoystickAttachVirtualInner(const SDL_VirtualJoystickDesc *desc);
extern bool SDL_JoystickDetachVirtualInner(SDL_JoystickID instance_id);

extern bool SDL_SetJoystickVirtualAxisInner(SDL_Joystick *joystick, int axis, Sint16 value);
extern bool SDL_SetJoystickVirtualBallInner(SDL_Joystick *joystick, int ball, Sint16 xrel, Sint16 yrel);
extern bool SDL_SetJoystickVirtualButtonInner(SDL_Joystick *joystick, int button, bool down);
extern bool SDL_SetJoystickVirtualHatInner(SDL_Joystick *joystick, int hat, Uint8 value);
extern bool SDL_SetJoystickVirtualTouchpadInner(SDL_Joystick *joystick, int touchpad, int finger, bool down, float x, float y, float pressure);
extern bool SDL_SendJoystickVirtualSensorDataInner(SDL_Joystick *joystick, SDL_SensorType type, Uint64 sensor_timestamp, const float *data, int num_values);

#endif // SDL_JOYSTICK_VIRTUAL

#endif // SDL_VIRTUALJOYSTICK_C_H
