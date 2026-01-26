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

#ifndef SDL_JOYSTICK_IOKIT_H
#define SDL_JOYSTICK_IOKIT_H

#include <IOKit/hid/IOHIDLib.h>
#include <ForceFeedback/ForceFeedback.h>
#include <ForceFeedback/ForceFeedbackConstants.h>

struct recElement
{
    IOHIDElementRef elementRef;
    IOHIDElementCookie cookie;
    uint32_t usagePage, usage; // HID usage
    SInt32 min;                // reported min value possible
    SInt32 max;                // reported max value possible

    // runtime variables used for auto-calibration
    SInt32 minReport; // min returned value
    SInt32 maxReport; // max returned value

    struct recElement *pNext; // next element in list
};
typedef struct recElement recElement;

struct joystick_hwdata
{
    IOHIDDeviceRef deviceRef; // HIDManager device handle
    io_service_t ffservice;   // Interface for force feedback, 0 = no ff
    FFDeviceObjectReference ffdevice;
    FFEFFECT *ffeffect;
    FFEffectObjectReference ffeffect_ref;
    bool ff_initialized;

    char product[256];  // name of product
    uint32_t usage;     // usage page from IOUSBHID Parser.h which defines general usage
    uint32_t usagePage; // usage within above page from IOUSBHID Parser.h which defines specific usage

    int axes;     // number of axis (calculated, not reported by device)
    int buttons;  // number of buttons (calculated, not reported by device)
    int hats;     // number of hat switches (calculated, not reported by device)
    int elements; // number of total elements (should be total of above) (calculated, not reported by device)

    recElement *firstAxis;
    recElement *firstButton;
    recElement *firstHat;

    bool removed;
    SDL_Joystick *joystick;
    bool runLoopAttached; // is 'deviceRef' attached to a CFRunLoop?

    int instance_id;
    SDL_GUID guid;
    int steam_virtual_gamepad_slot;
    bool nacon_revolution_x_unlimited;

    struct joystick_hwdata *pNext; // next device
};
typedef struct joystick_hwdata recDevice;

#endif // SDL_JOYSTICK_IOKIT_H
