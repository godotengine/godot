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

#include "SDL_evdev_capabilities.h"

#ifdef HAVE_LINUX_INPUT_H

// missing defines in older Linux kernel headers
#ifndef BTN_TRIGGER_HAPPY
#define BTN_TRIGGER_HAPPY 0x2c0
#endif
#ifndef BTN_DPAD_UP
#define BTN_DPAD_UP 0x220
#endif
#ifndef KEY_ALS_TOGGLE
#define KEY_ALS_TOGGLE 0x230
#endif

extern int
SDL_EVDEV_GuessDeviceClass(const unsigned long bitmask_props[NBITS(INPUT_PROP_MAX)],
                           const unsigned long bitmask_ev[NBITS(EV_MAX)],
                           const unsigned long bitmask_abs[NBITS(ABS_MAX)],
                           const unsigned long bitmask_key[NBITS(KEY_MAX)],
                           const unsigned long bitmask_rel[NBITS(REL_MAX)])
{
    struct range
    {
        unsigned start;
        unsigned end;
    };

    // key code ranges above BTN_MISC (start is inclusive, stop is exclusive)
    static const struct range high_key_blocks[] = {
        { KEY_OK, BTN_DPAD_UP },
        { KEY_ALS_TOGGLE, BTN_TRIGGER_HAPPY }
    };

    int devclass = 0;
    unsigned long keyboard_mask;

    // If the kernel specifically says it's an accelerometer, believe it
    if (test_bit(INPUT_PROP_ACCELEROMETER, bitmask_props)) {
        return SDL_UDEV_DEVICE_ACCELEROMETER;
    }

    // We treat pointing sticks as indistinguishable from mice
    if (test_bit(INPUT_PROP_POINTING_STICK, bitmask_props)) {
        return SDL_UDEV_DEVICE_MOUSE;
    }

    // We treat buttonpads as equivalent to touchpads
    if (test_bit(INPUT_PROP_TOPBUTTONPAD, bitmask_props) ||
        test_bit(INPUT_PROP_BUTTONPAD, bitmask_props) ||
        test_bit(INPUT_PROP_SEMI_MT, bitmask_props)) {
        return SDL_UDEV_DEVICE_TOUCHPAD;
    }

    // X, Y, Z axes but no buttons probably means an accelerometer
    if (test_bit(EV_ABS, bitmask_ev) &&
        test_bit(ABS_X, bitmask_abs) &&
        test_bit(ABS_Y, bitmask_abs) &&
        test_bit(ABS_Z, bitmask_abs) &&
        !test_bit(EV_KEY, bitmask_ev)) {
        return SDL_UDEV_DEVICE_ACCELEROMETER;
    }

    /* RX, RY, RZ axes but no buttons probably means a gyro or
     * accelerometer (we don't distinguish) */
    if (test_bit(EV_ABS, bitmask_ev) &&
        test_bit(ABS_RX, bitmask_abs) &&
        test_bit(ABS_RY, bitmask_abs) &&
        test_bit(ABS_RZ, bitmask_abs) &&
        !test_bit(EV_KEY, bitmask_ev)) {
        return SDL_UDEV_DEVICE_ACCELEROMETER;
    }

    if (test_bit(EV_ABS, bitmask_ev) &&
        test_bit(ABS_X, bitmask_abs) && test_bit(ABS_Y, bitmask_abs)) {
        if (test_bit(BTN_STYLUS, bitmask_key) || test_bit(BTN_TOOL_PEN, bitmask_key)) {
            ; // ID_INPUT_TABLET
        } else if (test_bit(BTN_TOOL_FINGER, bitmask_key) && !test_bit(BTN_TOOL_PEN, bitmask_key)) {
            devclass |= SDL_UDEV_DEVICE_TOUCHPAD; // ID_INPUT_TOUCHPAD
        } else if (test_bit(BTN_MOUSE, bitmask_key)) {
            devclass |= SDL_UDEV_DEVICE_MOUSE; // ID_INPUT_MOUSE
        } else if (test_bit(BTN_TOUCH, bitmask_key)) {
            /* TODO: better determining between touchscreen and multitouch touchpad,
               see https://github.com/systemd/systemd/blob/master/src/udev/udev-builtin-input_id.c */
            devclass |= SDL_UDEV_DEVICE_TOUCHSCREEN; // ID_INPUT_TOUCHSCREEN
        }

        if (test_bit(BTN_TRIGGER, bitmask_key) ||
            test_bit(BTN_A, bitmask_key) ||
            test_bit(BTN_1, bitmask_key) ||
            test_bit(ABS_RX, bitmask_abs) ||
            test_bit(ABS_RY, bitmask_abs) ||
            test_bit(ABS_RZ, bitmask_abs) ||
            test_bit(ABS_THROTTLE, bitmask_abs) ||
            test_bit(ABS_RUDDER, bitmask_abs) ||
            test_bit(ABS_WHEEL, bitmask_abs) ||
            test_bit(ABS_GAS, bitmask_abs) ||
            test_bit(ABS_BRAKE, bitmask_abs)) {
            devclass |= SDL_UDEV_DEVICE_JOYSTICK; // ID_INPUT_JOYSTICK
        }
    }

    if (test_bit(EV_REL, bitmask_ev) &&
        test_bit(REL_X, bitmask_rel) && test_bit(REL_Y, bitmask_rel) &&
        test_bit(BTN_MOUSE, bitmask_key)) {
        devclass |= SDL_UDEV_DEVICE_MOUSE; // ID_INPUT_MOUSE
    }

    if (test_bit(EV_KEY, bitmask_ev)) {
        unsigned i;
        unsigned long found = 0;

        for (i = 0; i < BTN_MISC / BITS_PER_LONG; ++i) {
            found |= bitmask_key[i];
        }
        // If there are no keys in the lower block, check the higher blocks
        if (!found) {
            unsigned block;
            for (block = 0; block < (sizeof(high_key_blocks) / sizeof(struct range)); ++block) {
                for (i = high_key_blocks[block].start; i < high_key_blocks[block].end; ++i) {
                    if (test_bit(i, bitmask_key)) {
                        found = 1;
                        break;
                    }
                }
            }
        }

        if (found > 0) {
            devclass |= SDL_UDEV_DEVICE_HAS_KEYS; // ID_INPUT_KEY
        }
    }

    /* the first 32 bits are ESC, numbers, and Q to D, so if we have all of
     * those, consider it to be a fully-featured keyboard;
     * do not test KEY_RESERVED, though */
    keyboard_mask = 0xFFFFFFFE;
    if ((bitmask_key[0] & keyboard_mask) == keyboard_mask) {
        devclass |= SDL_UDEV_DEVICE_KEYBOARD; // ID_INPUT_KEYBOARD
    }

    return devclass;
}

#endif // HAVE_LINUX_INPUT_H
