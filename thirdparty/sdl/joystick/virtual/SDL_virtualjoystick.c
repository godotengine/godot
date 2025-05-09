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

#ifdef SDL_JOYSTICK_VIRTUAL

// This is the virtual implementation of the SDL joystick API

#include "SDL_virtualjoystick_c.h"
#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"

static joystick_hwdata *g_VJoys SDL_GUARDED_BY(SDL_joystick_lock) = NULL;

static joystick_hwdata *VIRTUAL_HWDataForInstance(SDL_JoystickID instance_id)
{
    joystick_hwdata *vjoy;

    SDL_AssertJoysticksLocked();

    for (vjoy = g_VJoys; vjoy; vjoy = vjoy->next) {
        if (instance_id == vjoy->instance_id) {
            return vjoy;
        }
    }
    return NULL;
}

static joystick_hwdata *VIRTUAL_HWDataForIndex(int device_index)
{
    joystick_hwdata *vjoy;

    SDL_AssertJoysticksLocked();

    for (vjoy = g_VJoys; vjoy; vjoy = vjoy->next) {
        if (device_index == 0) {
            break;
        }
        --device_index;
    }
    return vjoy;
}

static void VIRTUAL_FreeHWData(joystick_hwdata *hwdata)
{
    joystick_hwdata *cur;
    joystick_hwdata *prev = NULL;

    SDL_AssertJoysticksLocked();

    if (!hwdata) {
        return;
    }

    if (hwdata->desc.Cleanup) {
        hwdata->desc.Cleanup(hwdata->desc.userdata);
    }

    // Remove hwdata from SDL-global list
    for (cur = g_VJoys; cur; prev = cur, cur = cur->next) {
        if (hwdata == cur) {
            if (prev) {
                prev->next = cur->next;
            } else {
                g_VJoys = cur->next;
            }
            break;
        }
    }

    if (hwdata->joystick) {
        hwdata->joystick->hwdata = NULL;
        hwdata->joystick = NULL;
    }
    if (hwdata->name) {
        SDL_free(hwdata->name);
        hwdata->name = NULL;
    }
    if (hwdata->axes) {
        SDL_free((void *)hwdata->axes);
        hwdata->axes = NULL;
    }
    if (hwdata->buttons) {
        SDL_free(hwdata->buttons);
        hwdata->buttons = NULL;
    }
    if (hwdata->hats) {
        SDL_free(hwdata->hats);
        hwdata->hats = NULL;
    }
    if (hwdata->balls) {
        SDL_free(hwdata->balls);
        hwdata->balls = NULL;
    }
    if (hwdata->touchpads) {
        for (Uint16 i = 0; i < hwdata->desc.ntouchpads; ++i) {
            SDL_free(hwdata->touchpads[i].fingers);
            hwdata->touchpads[i].fingers = NULL;
        }
        SDL_free(hwdata->touchpads);
        hwdata->touchpads = NULL;
    }
    if (hwdata->sensors) {
        SDL_free(hwdata->sensors);
        hwdata->sensors = NULL;
    }
    if (hwdata->sensor_events) {
        SDL_free(hwdata->sensor_events);
        hwdata->sensor_events = NULL;
    }
    SDL_free(hwdata);
}

SDL_JoystickID SDL_JoystickAttachVirtualInner(const SDL_VirtualJoystickDesc *desc)
{
    joystick_hwdata *hwdata = NULL;
    const char *name = NULL;
    int axis_triggerleft = -1;
    int axis_triggerright = -1;

    SDL_AssertJoysticksLocked();

    if (!desc) {
        SDL_InvalidParamError("desc");
        return 0;
    }
    if (desc->version < sizeof(*desc)) {
        // Update this to handle older versions of this interface
        SDL_SetError("Invalid desc, should be initialized with SDL_INIT_INTERFACE()");
        return 0;
    }

    hwdata = (joystick_hwdata *)SDL_calloc(1, sizeof(joystick_hwdata));
    if (!hwdata) {
        VIRTUAL_FreeHWData(hwdata);
        return 0;
    }
    SDL_copyp(&hwdata->desc, desc);
    hwdata->desc.touchpads = NULL;
    hwdata->desc.sensors = NULL;

    if (hwdata->desc.name) {
        name = hwdata->desc.name;
    } else {
        switch (hwdata->desc.type) {
        case SDL_JOYSTICK_TYPE_GAMEPAD:
            name = "Virtual Controller";
            break;
        case SDL_JOYSTICK_TYPE_WHEEL:
            name = "Virtual Wheel";
            break;
        case SDL_JOYSTICK_TYPE_ARCADE_STICK:
            name = "Virtual Arcade Stick";
            break;
        case SDL_JOYSTICK_TYPE_FLIGHT_STICK:
            name = "Virtual Flight Stick";
            break;
        case SDL_JOYSTICK_TYPE_DANCE_PAD:
            name = "Virtual Dance Pad";
            break;
        case SDL_JOYSTICK_TYPE_GUITAR:
            name = "Virtual Guitar";
            break;
        case SDL_JOYSTICK_TYPE_DRUM_KIT:
            name = "Virtual Drum Kit";
            break;
        case SDL_JOYSTICK_TYPE_ARCADE_PAD:
            name = "Virtual Arcade Pad";
            break;
        case SDL_JOYSTICK_TYPE_THROTTLE:
            name = "Virtual Throttle";
            break;
        default:
            name = "Virtual Joystick";
            break;
        }
    }
    hwdata->name = SDL_strdup(name);

    if (hwdata->desc.type == SDL_JOYSTICK_TYPE_GAMEPAD) {
        int i, axis;

        if (hwdata->desc.button_mask == 0) {
            for (i = 0; i < hwdata->desc.nbuttons && i < sizeof(hwdata->desc.button_mask) * 8; ++i) {
                hwdata->desc.button_mask |= (1 << i);
            }
        }

        if (hwdata->desc.axis_mask == 0) {
            if (hwdata->desc.naxes >= 2) {
                hwdata->desc.axis_mask |= ((1 << SDL_GAMEPAD_AXIS_LEFTX) | (1 << SDL_GAMEPAD_AXIS_LEFTY));
            }
            if (hwdata->desc.naxes >= 4) {
                hwdata->desc.axis_mask |= ((1 << SDL_GAMEPAD_AXIS_RIGHTX) | (1 << SDL_GAMEPAD_AXIS_RIGHTY));
            }
            if (hwdata->desc.naxes >= 6) {
                hwdata->desc.axis_mask |= ((1 << SDL_GAMEPAD_AXIS_LEFT_TRIGGER) | (1 << SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
            }
        }

        // Find the trigger axes
        axis = 0;
        for (i = 0; axis < hwdata->desc.naxes && i < SDL_GAMEPAD_AXIS_COUNT; ++i) {
            if (hwdata->desc.axis_mask & (1 << i)) {
                if (i == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) {
                    axis_triggerleft = axis;
                }
                if (i == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
                    axis_triggerright = axis;
                }
                ++axis;
            }
        }
    }

    hwdata->guid = SDL_CreateJoystickGUID(SDL_HARDWARE_BUS_VIRTUAL, hwdata->desc.vendor_id, hwdata->desc.product_id, 0, NULL, name, 'v', (Uint8)hwdata->desc.type);

    // Allocate fields for different control-types
    if (hwdata->desc.naxes > 0) {
        hwdata->axes = (Sint16 *)SDL_calloc(hwdata->desc.naxes, sizeof(*hwdata->axes));
        if (!hwdata->axes) {
            VIRTUAL_FreeHWData(hwdata);
            return 0;
        }

        // Trigger axes are at minimum value at rest
        if (axis_triggerleft >= 0) {
            hwdata->axes[axis_triggerleft] = SDL_JOYSTICK_AXIS_MIN;
        }
        if (axis_triggerright >= 0) {
            hwdata->axes[axis_triggerright] = SDL_JOYSTICK_AXIS_MIN;
        }
    }
    if (hwdata->desc.nbuttons > 0) {
        hwdata->buttons = (bool *)SDL_calloc(hwdata->desc.nbuttons, sizeof(*hwdata->buttons));
        if (!hwdata->buttons) {
            VIRTUAL_FreeHWData(hwdata);
            return 0;
        }
    }
    if (hwdata->desc.nhats > 0) {
        hwdata->hats = (Uint8 *)SDL_calloc(hwdata->desc.nhats, sizeof(*hwdata->hats));
        if (!hwdata->hats) {
            VIRTUAL_FreeHWData(hwdata);
            return 0;
        }
    }
    if (hwdata->desc.nballs > 0) {
        hwdata->balls = (SDL_JoystickBallData *)SDL_calloc(hwdata->desc.nballs, sizeof(*hwdata->balls));
        if (!hwdata->balls) {
            VIRTUAL_FreeHWData(hwdata);
            return 0;
        }
    }
    if (hwdata->desc.ntouchpads > 0) {
        if (!desc->touchpads) {
            VIRTUAL_FreeHWData(hwdata);
            SDL_SetError("desc missing touchpad descriptions");
            return 0;
        }
        hwdata->touchpads = (SDL_JoystickTouchpadInfo *)SDL_calloc(hwdata->desc.ntouchpads, sizeof(*hwdata->touchpads));
        if (!hwdata->touchpads) {
            VIRTUAL_FreeHWData(hwdata);
            return 0;
        }
        for (Uint16 i = 0; i < hwdata->desc.ntouchpads; ++i) {
            const SDL_VirtualJoystickTouchpadDesc *touchpad_desc = &desc->touchpads[i];
            hwdata->touchpads[i].nfingers = touchpad_desc->nfingers;
            hwdata->touchpads[i].fingers = (SDL_JoystickTouchpadFingerInfo *)SDL_calloc(touchpad_desc->nfingers, sizeof(*hwdata->touchpads[i].fingers));
            if (!hwdata->touchpads[i].fingers) {
                VIRTUAL_FreeHWData(hwdata);
                return 0;
            }
        }
    }
    if (hwdata->desc.nsensors > 0) {
        if (!desc->sensors) {
            VIRTUAL_FreeHWData(hwdata);
            SDL_SetError("desc missing sensor descriptions");
            return 0;
        }
        hwdata->sensors = (SDL_JoystickSensorInfo *)SDL_calloc(hwdata->desc.nsensors, sizeof(*hwdata->sensors));
        if (!hwdata->sensors) {
            VIRTUAL_FreeHWData(hwdata);
            return 0;
        }
        for (Uint16 i = 0; i < hwdata->desc.nsensors; ++i) {
            const SDL_VirtualJoystickSensorDesc *sensor_desc = &desc->sensors[i];
            hwdata->sensors[i].type = sensor_desc->type;
            hwdata->sensors[i].rate = sensor_desc->rate;
        }
    }

    // Allocate an instance ID for this device
    hwdata->instance_id = SDL_GetNextObjectID();

    // Add virtual joystick to SDL-global lists
    if (g_VJoys) {
        joystick_hwdata *last;

        for (last = g_VJoys; last->next; last = last->next) {
        }
        last->next = hwdata;
    } else {
        g_VJoys = hwdata;
    }
    SDL_PrivateJoystickAdded(hwdata->instance_id);

    return hwdata->instance_id;
}

bool SDL_JoystickDetachVirtualInner(SDL_JoystickID instance_id)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForInstance(instance_id);
    if (!hwdata) {
        return SDL_SetError("Virtual joystick data not found");
    }
    VIRTUAL_FreeHWData(hwdata);
    SDL_PrivateJoystickRemoved(instance_id);
    return true;
}

bool SDL_SetJoystickVirtualAxisInner(SDL_Joystick *joystick, int axis, Sint16 value)
{
    joystick_hwdata *hwdata;

    SDL_AssertJoysticksLocked();

    if (!joystick || !joystick->hwdata) {
        return SDL_SetError("Invalid joystick");
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;
    if (axis < 0 || axis >= hwdata->desc.naxes) {
        return SDL_SetError("Invalid axis index");
    }

    hwdata->axes[axis] = value;
    hwdata->changes |= AXES_CHANGED;

    return true;
}

bool SDL_SetJoystickVirtualBallInner(SDL_Joystick *joystick, int ball, Sint16 xrel, Sint16 yrel)
{
    joystick_hwdata *hwdata;

    SDL_AssertJoysticksLocked();

    if (!joystick || !joystick->hwdata) {
        return SDL_SetError("Invalid joystick");
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;
    if (ball < 0 || ball >= hwdata->desc.nballs) {
        return SDL_SetError("Invalid ball index");
    }

    hwdata->balls[ball].dx += xrel;
    hwdata->balls[ball].dx = SDL_clamp(hwdata->balls[ball].dx, SDL_MIN_SINT16, SDL_MAX_SINT16);
    hwdata->balls[ball].dy += yrel;
    hwdata->balls[ball].dy = SDL_clamp(hwdata->balls[ball].dy, SDL_MIN_SINT16, SDL_MAX_SINT16);
    hwdata->changes |= BALLS_CHANGED;

    return true;
}

bool SDL_SetJoystickVirtualButtonInner(SDL_Joystick *joystick, int button, bool down)
{
    joystick_hwdata *hwdata;

    SDL_AssertJoysticksLocked();

    if (!joystick || !joystick->hwdata) {
        return SDL_SetError("Invalid joystick");
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;
    if (button < 0 || button >= hwdata->desc.nbuttons) {
        return SDL_SetError("Invalid button index");
    }

    hwdata->buttons[button] = down;
    hwdata->changes |= BUTTONS_CHANGED;

    return true;
}

bool SDL_SetJoystickVirtualHatInner(SDL_Joystick *joystick, int hat, Uint8 value)
{
    joystick_hwdata *hwdata;

    SDL_AssertJoysticksLocked();

    if (!joystick || !joystick->hwdata) {
        return SDL_SetError("Invalid joystick");
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;
    if (hat < 0 || hat >= hwdata->desc.nhats) {
        return SDL_SetError("Invalid hat index");
    }

    hwdata->hats[hat] = value;
    hwdata->changes |= HATS_CHANGED;

    return true;
}

bool SDL_SetJoystickVirtualTouchpadInner(SDL_Joystick *joystick, int touchpad, int finger, bool down, float x, float y, float pressure)
{
    joystick_hwdata *hwdata;

    SDL_AssertJoysticksLocked();

    if (!joystick || !joystick->hwdata) {
        return SDL_SetError("Invalid joystick");
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;
    if (touchpad < 0 || touchpad >= hwdata->desc.ntouchpads) {
        return SDL_SetError("Invalid touchpad index");
    }
    if (finger < 0 || finger >= hwdata->touchpads[touchpad].nfingers) {
        return SDL_SetError("Invalid finger index");
    }

    SDL_JoystickTouchpadFingerInfo *info = &hwdata->touchpads[touchpad].fingers[finger];
    info->down = down;
    info->x = x;
    info->y = y;
    info->pressure = pressure;
    hwdata->changes |= TOUCHPADS_CHANGED;

    return true;
}

bool SDL_SendJoystickVirtualSensorDataInner(SDL_Joystick *joystick, SDL_SensorType type, Uint64 sensor_timestamp, const float *data, int num_values)
{
    joystick_hwdata *hwdata;

    SDL_AssertJoysticksLocked();

    if (!joystick || !joystick->hwdata) {
        return SDL_SetError("Invalid joystick");
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;
    if (hwdata->num_sensor_events == hwdata->max_sensor_events) {
        int new_max_sensor_events = (hwdata->max_sensor_events + 1);
        VirtualSensorEvent *sensor_events = (VirtualSensorEvent *)SDL_realloc(hwdata->sensor_events, new_max_sensor_events * sizeof(*sensor_events));
        if (!sensor_events) {
            return false;
        }
        hwdata->sensor_events = sensor_events;
        hwdata->max_sensor_events = hwdata->max_sensor_events;
    }

    VirtualSensorEvent *event = &hwdata->sensor_events[hwdata->num_sensor_events++];
    event->type = type;
    event->sensor_timestamp = sensor_timestamp;
    event->num_values = SDL_min(num_values, SDL_arraysize(event->data));
    SDL_memcpy(event->data, data, (event->num_values * sizeof(*event->data)));

    return true;
}

static bool VIRTUAL_JoystickInit(void)
{
    return true;
}

static int VIRTUAL_JoystickGetCount(void)
{
    joystick_hwdata *cur;
    int count = 0;

    SDL_AssertJoysticksLocked();

    for (cur = g_VJoys; cur; cur = cur->next) {
        ++count;
    }
    return count;
}

static void VIRTUAL_JoystickDetect(void)
{
}

static bool VIRTUAL_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    // We don't override any other drivers... or do we?
    return false;
}

static const char *VIRTUAL_JoystickGetDeviceName(int device_index)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);
    if (!hwdata) {
        return NULL;
    }
    return hwdata->name;
}

static const char *VIRTUAL_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int VIRTUAL_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int VIRTUAL_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void VIRTUAL_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);

    if (hwdata && hwdata->desc.SetPlayerIndex) {
        hwdata->desc.SetPlayerIndex(hwdata->desc.userdata, player_index);
    }
}

static SDL_GUID VIRTUAL_JoystickGetDeviceGUID(int device_index)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);
    if (!hwdata) {
        SDL_GUID guid;
        SDL_zero(guid);
        return guid;
    }
    return hwdata->guid;
}

static SDL_JoystickID VIRTUAL_JoystickGetDeviceInstanceID(int device_index)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);
    if (!hwdata) {
        return true;
    }
    return hwdata->instance_id;
}

static bool VIRTUAL_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    joystick_hwdata *hwdata;

    SDL_AssertJoysticksLocked();

    hwdata = VIRTUAL_HWDataForIndex(device_index);
    if (!hwdata) {
        return SDL_SetError("No such device");
    }
    joystick->hwdata = hwdata;
    joystick->naxes = hwdata->desc.naxes;
    joystick->nbuttons = hwdata->desc.nbuttons;
    joystick->nhats = hwdata->desc.nhats;
    hwdata->joystick = joystick;

    for (Uint16 i = 0; i < hwdata->desc.ntouchpads; ++i) {
        const SDL_JoystickTouchpadInfo *touchpad = &hwdata->touchpads[i];
        SDL_PrivateJoystickAddTouchpad(joystick, touchpad->nfingers);
    }
    for (Uint16 i = 0; i < hwdata->desc.nsensors; ++i) {
        const SDL_JoystickSensorInfo *sensor = &hwdata->sensors[i];
        SDL_PrivateJoystickAddSensor(joystick, sensor->type, sensor->rate);
    }

    if (hwdata->desc.SetLED) {
        SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_RGB_LED_BOOLEAN, true);
    }
    if (hwdata->desc.Rumble) {
        SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_RUMBLE_BOOLEAN, true);
    }
    if (hwdata->desc.RumbleTriggers) {
        SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_TRIGGER_RUMBLE_BOOLEAN, true);
    }
    return true;
}

static bool VIRTUAL_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    bool result;

    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        if (hwdata->desc.Rumble) {
            result = hwdata->desc.Rumble(hwdata->desc.userdata, low_frequency_rumble, high_frequency_rumble);
        } else {
            result = SDL_Unsupported();
        }
    } else {
        result = SDL_SetError("Rumble failed, device disconnected");
    }

    return result;
}

static bool VIRTUAL_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    bool result;

    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        if (hwdata->desc.RumbleTriggers) {
            result = hwdata->desc.RumbleTriggers(hwdata->desc.userdata, left_rumble, right_rumble);
        } else {
            result = SDL_Unsupported();
        }
    } else {
        result = SDL_SetError("Rumble failed, device disconnected");
    }

    return result;
}

static bool VIRTUAL_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    bool result;

    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        if (hwdata->desc.SetLED) {
            result = hwdata->desc.SetLED(hwdata->desc.userdata, red, green, blue);
        } else {
            result = SDL_Unsupported();
        }
    } else {
        result = SDL_SetError("SetLED failed, device disconnected");
    }

    return result;
}

static bool VIRTUAL_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    bool result;

    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        if (hwdata->desc.SendEffect) {
            result = hwdata->desc.SendEffect(hwdata->desc.userdata, data, size);
        } else {
            result = SDL_Unsupported();
        }
    } else {
        result = SDL_SetError("SendEffect failed, device disconnected");
    }

    return result;
}

static bool VIRTUAL_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    bool result;

    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        if (hwdata->desc.SetSensorsEnabled) {
            result = hwdata->desc.SetSensorsEnabled(hwdata->desc.userdata, enabled);
        } else {
            result = true;
        }
        if (result) {
            hwdata->sensors_enabled = enabled;
        }
    } else {
        result = SDL_SetError("SetSensorsEnabled failed, device disconnected");
    }

    return result;
}

static void VIRTUAL_JoystickUpdate(SDL_Joystick *joystick)
{
    joystick_hwdata *hwdata;
    Uint64 timestamp = SDL_GetTicksNS();

    SDL_AssertJoysticksLocked();

    if (!joystick) {
        return;
    }
    if (!joystick->hwdata) {
        return;
    }

    hwdata = (joystick_hwdata *)joystick->hwdata;

    if (hwdata->desc.Update) {
        hwdata->desc.Update(hwdata->desc.userdata);
    }

    if (hwdata->changes & AXES_CHANGED) {
        for (Uint8 i = 0; i < hwdata->desc.naxes; ++i) {
            SDL_SendJoystickAxis(timestamp, joystick, i, hwdata->axes[i]);
        }
    }
    if (hwdata->changes & BALLS_CHANGED) {
        for (Uint8 i = 0; i < hwdata->desc.nballs; ++i) {
            SDL_JoystickBallData *ball = &hwdata->balls[i];
            if (ball->dx || ball->dy) {
                SDL_SendJoystickBall(timestamp, joystick, i, (Sint16)ball->dx, (Sint16)ball->dy);
                ball->dx = 0;
                ball->dy = 0;
            }
        }
    }
    if (hwdata->changes & BUTTONS_CHANGED) {
        for (Uint8 i = 0; i < hwdata->desc.nbuttons; ++i) {
            SDL_SendJoystickButton(timestamp, joystick, i, hwdata->buttons[i]);
        }
    }
    if (hwdata->changes & HATS_CHANGED) {
        for (Uint8 i = 0; i < hwdata->desc.nhats; ++i) {
            SDL_SendJoystickHat(timestamp, joystick, i, hwdata->hats[i]);
        }
    }
    if (hwdata->changes & TOUCHPADS_CHANGED) {
        for (Uint16 i = 0; i < hwdata->desc.ntouchpads; ++i) {
            const SDL_JoystickTouchpadInfo *touchpad = &hwdata->touchpads[i];
            for (int j = 0; j < touchpad->nfingers; ++j) {
                const SDL_JoystickTouchpadFingerInfo *finger = &touchpad->fingers[j];
                SDL_SendJoystickTouchpad(timestamp, joystick, i, j, finger->down, finger->x, finger->y, finger->pressure);
            }
        }
    }
    if (hwdata->num_sensor_events > 0) {
        if (hwdata->sensors_enabled) {
            for (int i = 0; i < hwdata->num_sensor_events; ++i) {
                const VirtualSensorEvent *event = &hwdata->sensor_events[i];
                SDL_SendJoystickSensor(timestamp, joystick, event->type, event->sensor_timestamp, event->data, event->num_values);
            }
        }
        hwdata->num_sensor_events = 0;
    }
    hwdata->changes = 0;
}

static void VIRTUAL_JoystickClose(SDL_Joystick *joystick)
{
    SDL_AssertJoysticksLocked();

    if (joystick->hwdata) {
        joystick_hwdata *hwdata = joystick->hwdata;
        hwdata->joystick = NULL;
        joystick->hwdata = NULL;
    }
}

static void VIRTUAL_JoystickQuit(void)
{
    SDL_AssertJoysticksLocked();

    while (g_VJoys) {
        VIRTUAL_FreeHWData(g_VJoys);
    }
}

static bool VIRTUAL_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    joystick_hwdata *hwdata = VIRTUAL_HWDataForIndex(device_index);
    Uint8 current_button = 0;
    Uint8 current_axis = 0;

    if (!hwdata || hwdata->desc.type != SDL_JOYSTICK_TYPE_GAMEPAD) {
        return false;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_SOUTH))) {
        out->a.kind = EMappingKind_Button;
        out->a.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_EAST))) {
        out->b.kind = EMappingKind_Button;
        out->b.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_WEST))) {
        out->x.kind = EMappingKind_Button;
        out->x.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_NORTH))) {
        out->y.kind = EMappingKind_Button;
        out->y.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_BACK))) {
        out->back.kind = EMappingKind_Button;
        out->back.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_GUIDE))) {
        out->guide.kind = EMappingKind_Button;
        out->guide.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_START))) {
        out->start.kind = EMappingKind_Button;
        out->start.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_LEFT_STICK))) {
        out->leftstick.kind = EMappingKind_Button;
        out->leftstick.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_RIGHT_STICK))) {
        out->rightstick.kind = EMappingKind_Button;
        out->rightstick.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_LEFT_SHOULDER))) {
        out->leftshoulder.kind = EMappingKind_Button;
        out->leftshoulder.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER))) {
        out->rightshoulder.kind = EMappingKind_Button;
        out->rightshoulder.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_DPAD_UP))) {
        out->dpup.kind = EMappingKind_Button;
        out->dpup.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_DPAD_DOWN))) {
        out->dpdown.kind = EMappingKind_Button;
        out->dpdown.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_DPAD_LEFT))) {
        out->dpleft.kind = EMappingKind_Button;
        out->dpleft.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_DPAD_RIGHT))) {
        out->dpright.kind = EMappingKind_Button;
        out->dpright.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_MISC1))) {
        out->misc1.kind = EMappingKind_Button;
        out->misc1.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1))) {
        out->right_paddle1.kind = EMappingKind_Button;
        out->right_paddle1.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_LEFT_PADDLE1))) {
        out->left_paddle1.kind = EMappingKind_Button;
        out->left_paddle1.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_RIGHT_PADDLE2))) {
        out->right_paddle2.kind = EMappingKind_Button;
        out->right_paddle2.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_LEFT_PADDLE2))) {
        out->left_paddle2.kind = EMappingKind_Button;
        out->left_paddle2.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_TOUCHPAD))) {
        out->touchpad.kind = EMappingKind_Button;
        out->touchpad.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_MISC2))) {
        out->misc2.kind = EMappingKind_Button;
        out->misc2.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_MISC3))) {
        out->misc3.kind = EMappingKind_Button;
        out->misc3.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_MISC4))) {
        out->misc4.kind = EMappingKind_Button;
        out->misc4.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_MISC5))) {
        out->misc5.kind = EMappingKind_Button;
        out->misc5.target = current_button++;
    }

    if (current_button < hwdata->desc.nbuttons && (hwdata->desc.button_mask & (1 << SDL_GAMEPAD_BUTTON_MISC6))) {
        out->misc6.kind = EMappingKind_Button;
        out->misc6.target = current_button++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_GAMEPAD_AXIS_LEFTX))) {
        out->leftx.kind = EMappingKind_Axis;
        out->leftx.target = current_axis++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_GAMEPAD_AXIS_LEFTY))) {
        out->lefty.kind = EMappingKind_Axis;
        out->lefty.target = current_axis++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_GAMEPAD_AXIS_RIGHTX))) {
        out->rightx.kind = EMappingKind_Axis;
        out->rightx.target = current_axis++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_GAMEPAD_AXIS_RIGHTY))) {
        out->righty.kind = EMappingKind_Axis;
        out->righty.target = current_axis++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_GAMEPAD_AXIS_LEFT_TRIGGER))) {
        out->lefttrigger.kind = EMappingKind_Axis;
        out->lefttrigger.target = current_axis++;
    }

    if (current_axis < hwdata->desc.naxes && (hwdata->desc.axis_mask & (1 << SDL_GAMEPAD_AXIS_RIGHT_TRIGGER))) {
        out->righttrigger.kind = EMappingKind_Axis;
        out->righttrigger.target = current_axis++;
    }

    return true;
}

SDL_JoystickDriver SDL_VIRTUAL_JoystickDriver = {
    VIRTUAL_JoystickInit,
    VIRTUAL_JoystickGetCount,
    VIRTUAL_JoystickDetect,
    VIRTUAL_JoystickIsDevicePresent,
    VIRTUAL_JoystickGetDeviceName,
    VIRTUAL_JoystickGetDevicePath,
    VIRTUAL_JoystickGetDeviceSteamVirtualGamepadSlot,
    VIRTUAL_JoystickGetDevicePlayerIndex,
    VIRTUAL_JoystickSetDevicePlayerIndex,
    VIRTUAL_JoystickGetDeviceGUID,
    VIRTUAL_JoystickGetDeviceInstanceID,
    VIRTUAL_JoystickOpen,
    VIRTUAL_JoystickRumble,
    VIRTUAL_JoystickRumbleTriggers,
    VIRTUAL_JoystickSetLED,
    VIRTUAL_JoystickSendEffect,
    VIRTUAL_JoystickSetSensorsEnabled,
    VIRTUAL_JoystickUpdate,
    VIRTUAL_JoystickClose,
    VIRTUAL_JoystickQuit,
    VIRTUAL_JoystickGetGamepadMapping
};

#endif // SDL_JOYSTICK_VIRTUAL
