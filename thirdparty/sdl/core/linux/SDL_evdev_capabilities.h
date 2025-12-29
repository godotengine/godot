/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>
  Copyright (C) 2020 Collabora Ltd.

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

#ifndef SDL_evdev_capabilities_h_
#define SDL_evdev_capabilities_h_

#ifdef HAVE_LINUX_INPUT_H

#include <linux/input.h>

#ifndef INPUT_PROP_SEMI_MT
#define INPUT_PROP_SEMI_MT          0x03
#endif
#ifndef INPUT_PROP_TOPBUTTONPAD
#define INPUT_PROP_TOPBUTTONPAD     0x04
#endif
#ifndef INPUT_PROP_POINTING_STICK
#define INPUT_PROP_POINTING_STICK   0x05
#endif
#ifndef INPUT_PROP_ACCELEROMETER
#define INPUT_PROP_ACCELEROMETER    0x06
#endif
#ifndef INPUT_PROP_MAX
#define INPUT_PROP_MAX 0x1f
#endif

// A device can be any combination of these classes
typedef enum
{
    SDL_UDEV_DEVICE_UNKNOWN = 0x0000,
    SDL_UDEV_DEVICE_MOUSE = 0x0001,
    SDL_UDEV_DEVICE_KEYBOARD = 0x0002,
    SDL_UDEV_DEVICE_JOYSTICK = 0x0004,
    SDL_UDEV_DEVICE_SOUND = 0x0008,
    SDL_UDEV_DEVICE_TOUCHSCREEN = 0x0010,
    SDL_UDEV_DEVICE_ACCELEROMETER = 0x0020,
    SDL_UDEV_DEVICE_TOUCHPAD = 0x0040,
    SDL_UDEV_DEVICE_HAS_KEYS = 0x0080,
    SDL_UDEV_DEVICE_VIDEO_CAPTURE = 0x0100,
} SDL_UDEV_deviceclass;

#define BITS_PER_LONG        (sizeof(unsigned long) * 8)
#define NBITS(x)             ((((x)-1) / BITS_PER_LONG) + 1)
#define EVDEV_OFF(x)         ((x) % BITS_PER_LONG)
#define EVDEV_LONG(x)        ((x) / BITS_PER_LONG)
#define test_bit(bit, array) ((array[EVDEV_LONG(bit)] >> EVDEV_OFF(bit)) & 1)

extern int SDL_EVDEV_GuessDeviceClass(const unsigned long bitmask_props[NBITS(INPUT_PROP_MAX)],
                                      const unsigned long bitmask_ev[NBITS(EV_MAX)],
                                      const unsigned long bitmask_abs[NBITS(ABS_MAX)],
                                      const unsigned long bitmask_key[NBITS(KEY_MAX)],
                                      const unsigned long bitmask_rel[NBITS(REL_MAX)]);

#endif // HAVE_LINUX_INPUT_H

#endif // SDL_evdev_capabilities_h_
