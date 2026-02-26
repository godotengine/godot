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

#ifndef SDL_sysjoystick_c_h_
#define SDL_sysjoystick_c_h_

#include <linux/input.h>

struct SDL_joylist_item;
struct SDL_sensorlist_item;

// The private structure used to keep track of a joystick
struct joystick_hwdata
{
    int fd;
    // linux driver creates a separate device for gyro/accelerometer
    int fd_sensor;
    struct SDL_joylist_item *item;
    struct SDL_sensorlist_item *item_sensor;
    SDL_GUID guid;
    char *fname; // Used in haptic subsystem

    bool ff_rumble;
    bool ff_sine;
    struct ff_effect effect;
    Uint32 effect_expiration;

    // The current Linux joystick driver maps balls to two axes
    struct hwdata_ball
    {
        int axis[2];
    } *balls;

    // The current Linux joystick driver maps hats to two axes
    struct hwdata_hat
    {
        int axis[2];
    } *hats;

    // Support for the Linux 2.4 unified input interface
    Uint8 key_map[KEY_MAX];
    Uint8 abs_map[ABS_MAX];
    bool has_key[KEY_MAX];
    bool has_abs[ABS_MAX];
    bool has_accelerometer;
    bool has_gyro;

    // Support for the classic joystick interface
    bool classic;
    Uint16 *key_pam;
    Uint8 *abs_pam;

    struct axis_correct
    {
        bool use_deadzones;

        // Deadzone coefficients
        int coef[3];

        // Raw coordinate scale
        int minimum;
        int maximum;
        float scale;
    } abs_correct[ABS_MAX];

    float accelerometer_scale[3];
    float gyro_scale[3];

    /* Each axis is read independently, if we don't get all axis this call to
     * LINUX_JoystickUpdateupdate(), store them for the next one */
    float gyro_data[3];
    float accel_data[3];
    Uint64 sensor_tick;
    Sint32 last_tick;

    bool report_sensor;
    bool fresh;
    bool recovering_from_dropped;
    bool recovering_from_dropped_sensor;

    // Steam Controller support
    bool m_bSteamController;

    // 4 = (ABS_HAT3X-ABS_HAT0X)/2 (see input-event-codes.h in kernel)
    int hats_indices[4];
    bool has_hat[4];
    struct hat_axis_correct
    {
        bool use_deadzones;
        int minimum[2];
        int maximum[2];
    } hat_correct[4];

    // Set when gamepad is pending removal due to ENODEV read error
    bool gone;
    bool sensor_gone;
};

#endif // SDL_sysjoystick_c_h_
