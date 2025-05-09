/*
  Simple DirectMedia Layer
  Copyright (C) 2025 Simon Wood <simon@mungewell.org>
  Copyright (C) 2025 Michal Malý <madcatxster@devoid-pointer.net>
  Copyright (C) 2025 Katharine Chui <katharine.chui@gmail.com>

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

#ifdef SDL_JOYSTICK_HIDAPI

#include "../SDL_sysjoystick.h"
#include "SDL3/SDL_events.h"
#include "SDL_hidapijoystick_c.h"

#ifdef SDL_JOYSTICK_HIDAPI_LG4FF

#define USB_VENDOR_ID_LOGITECH 0x046d
#define USB_DEVICE_ID_LOGITECH_G29_WHEEL 0xc24f
#define USB_DEVICE_ID_LOGITECH_G27_WHEEL 0xc29b
#define USB_DEVICE_ID_LOGITECH_G25_WHEEL 0xc299
#define USB_DEVICE_ID_LOGITECH_DFGT_WHEEL 0xc29a
#define USB_DEVICE_ID_LOGITECH_DFP_WHEEL 0xc298
#define USB_DEVICE_ID_LOGITECH_WHEEL 0xc294

static Uint32 supported_device_ids[] = {
    USB_DEVICE_ID_LOGITECH_G29_WHEEL,
    USB_DEVICE_ID_LOGITECH_G27_WHEEL,
    USB_DEVICE_ID_LOGITECH_G25_WHEEL,
    USB_DEVICE_ID_LOGITECH_DFGT_WHEEL,
    USB_DEVICE_ID_LOGITECH_DFP_WHEEL,
    USB_DEVICE_ID_LOGITECH_WHEEL
};

// keep the same order as the supported_ids array
static const char *supported_device_names[] = {
    "Logitech G29",
    "Logitech G27",
    "Logitech G25",
    "Logitech Driving Force GT",
    "Logitech Driving Force Pro",
    "Driving Force EX"
};

static const char *HIDAPI_DriverLg4ff_GetDeviceName(Uint32 device_id)
{
    for (int i = 0;i < (sizeof supported_device_ids) / sizeof(Uint32);i++) {
        if (supported_device_ids[i] == device_id) {
            return supported_device_names[i];
        }
    }
    SDL_assert(0);
    return "";
}

static int HIDAPI_DriverLg4ff_GetNumberOfButtons(Uint32 device_id)
{
    switch (device_id) {
        case USB_DEVICE_ID_LOGITECH_G29_WHEEL:
            return 25;
        case USB_DEVICE_ID_LOGITECH_G27_WHEEL:
            return 22;
        case USB_DEVICE_ID_LOGITECH_G25_WHEEL:
            return 19;
        case USB_DEVICE_ID_LOGITECH_DFGT_WHEEL:
            return 21;
        case USB_DEVICE_ID_LOGITECH_DFP_WHEEL:
            return 14;
        case USB_DEVICE_ID_LOGITECH_WHEEL:
            return 13;
        default:
            SDL_assert(0);
            return 0;
    }
}

typedef struct
{
    Uint8 last_report_buf[32];
    bool initialized;
    bool is_ffex;
    Uint16 range;
} SDL_DriverLg4ff_Context;

static void HIDAPI_DriverLg4ff_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_LG4FF, callback, userdata);
}

static void HIDAPI_DriverLg4ff_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_LG4FF, callback, userdata);
}

static bool HIDAPI_DriverLg4ff_IsEnabled(void)
{
    bool enabled = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_LG4FF,
                              SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));

    return enabled;
}

/*
  Wheel id information by:
  Michal Malý <madcatxster@devoid-pointer.net> <madcatxster@gmail.com>
  Simon Wood <simon@mungewell.org>
  `git blame v6.12 drivers/hid/hid-lg4ff.c`, https://github.com/torvalds/linux.git
*/
static Uint16 HIDAPI_DriverLg4ff_IdentifyWheel(Uint16 device_id, Uint16 release_number)
{
    #define is_device(ret, m, r) { \
        if ((release_number & m) == r) { \
            return ret; \
        } \
    }
    #define is_dfp { \
        is_device(USB_DEVICE_ID_LOGITECH_DFP_WHEEL, 0xf000, 0x1000); \
    }
    #define is_dfgt { \
        is_device(USB_DEVICE_ID_LOGITECH_DFGT_WHEEL, 0xff00, 0x1300); \
    }
    #define is_g25 { \
        is_device(USB_DEVICE_ID_LOGITECH_G25_WHEEL, 0xff00, 0x1200); \
    }
    #define is_g27 { \
        is_device(USB_DEVICE_ID_LOGITECH_G27_WHEEL, 0xfff0, 0x1230); \
    }
    #define is_g29 { \
        is_device(USB_DEVICE_ID_LOGITECH_G29_WHEEL, 0xfff8, 0x1350); \
        is_device(USB_DEVICE_ID_LOGITECH_G29_WHEEL, 0xff00, 0x8900); \
    }
    switch(device_id){
        case USB_DEVICE_ID_LOGITECH_DFP_WHEEL:
        case USB_DEVICE_ID_LOGITECH_WHEEL:
            is_g29;
            is_g27;
            is_g25;
            is_dfgt;
            is_dfp;
            break;
        case USB_DEVICE_ID_LOGITECH_DFGT_WHEEL:
            is_g29;
            is_dfgt;
            break;
        case USB_DEVICE_ID_LOGITECH_G25_WHEEL:
            is_g29;
            is_g27;
            is_g25;
            break;
        case USB_DEVICE_ID_LOGITECH_G27_WHEEL:
            is_g29;
            is_g27;
            break;
        case USB_DEVICE_ID_LOGITECH_G29_WHEEL:
            is_g29;
            break;
    }
    return 0;
    #undef is_device
    #undef is_dfp
    #undef is_dfgt
    #undef is_g25
    #undef is_g27
    #undef is_g29
}

static int SDL_HIDAPI_DriverLg4ff_GetEnvInt(const char *env_name, int min, int max, int def)
{
    const char *env = SDL_getenv(env_name);
    int value = 0;
    if(env == NULL) {
        return def;
    }
    value = SDL_atoi(env);
    if (value < min) {
        value = min;
    }
    if (value > max) {
        value = max;
    }
    return value;
}

/*
  Commands by:
  Michal Malý <madcatxster@devoid-pointer.net> <madcatxster@gmail.com>
  Simon Wood <simon@mungewell.org>
  `git blame v6.12 drivers/hid/hid-lg4ff.c`, https://github.com/torvalds/linux.git
*/
static bool HIDAPI_DriverLg4ff_SwitchMode(SDL_HIDAPI_Device *device, Uint16 target_product_id){
    int ret = 0;

    switch(target_product_id){
        case USB_DEVICE_ID_LOGITECH_G29_WHEEL:{
            Uint8 cmd[] = {0xf8, 0x09, 0x05, 0x01, 0x01, 0x00, 0x00};
            ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
            break;
        }
        case USB_DEVICE_ID_LOGITECH_G27_WHEEL:{
            Uint8 cmd[] = {0xf8, 0x09, 0x04, 0x01, 0x00, 0x00, 0x00};
            ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
            break;
        }
        case USB_DEVICE_ID_LOGITECH_G25_WHEEL:{
            Uint8 cmd[] = {0xf8, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00};
            ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
            break;
        }
        case USB_DEVICE_ID_LOGITECH_DFGT_WHEEL:{
            Uint8 cmd[] = {0xf8, 0x09, 0x03, 0x01, 0x00, 0x00, 0x00};
            ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
            break;
        }
        case USB_DEVICE_ID_LOGITECH_DFP_WHEEL:{
            Uint8 cmd[] = {0xf8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00};
            ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
            break;
        }
        case USB_DEVICE_ID_LOGITECH_WHEEL:{
            Uint8 cmd[] = {0xf8, 0x09, 0x00, 0x01, 0x00, 0x00, 0x00};
            ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
            break;
        }
        default:{
            SDL_assert(0);
        }
    }
    if(ret == -1){
        return false;
    }
    return true;
}

static bool HIDAPI_DriverLg4ff_IsSupportedDevice(
    SDL_HIDAPI_Device *device,
    const char *name,
    SDL_GamepadType type,
    Uint16 vendor_id,
    Uint16 product_id,
    Uint16 version,
    int interface_number,
    int interface_class,
    int interface_subclass,
    int interface_protocol)
{
    int i;
    if (vendor_id != USB_VENDOR_ID_LOGITECH) {
        return false;
    }
    for (i = 0;i < sizeof(supported_device_ids) / sizeof(Uint32);i++) {
        if (supported_device_ids[i] == product_id) {
            break;
        }
    }
    if (i == sizeof(supported_device_ids) / sizeof(Uint32)) {
        return false;
    }
    Uint16 real_id = HIDAPI_DriverLg4ff_IdentifyWheel(product_id, version);
    if (real_id == product_id || real_id == 0) {
        // either it is already in native mode, or we don't know what the native mode is
        return true;
    }
    // a supported native mode is found, send mode change command, then still state that we support the device
    if (device != NULL && SDL_HIDAPI_DriverLg4ff_GetEnvInt("SDL_HIDAPI_LG4FF_NO_MODE_SWITCH", 0, 1, 0) == 0) {
        HIDAPI_DriverLg4ff_SwitchMode(device, real_id);
    }
    return true;
}

/*
  *Ported*
  Original functions by:
  Michal Malý <madcatxster@devoid-pointer.net> <madcatxster@gmail.com>
  lg4ff_set_range_g25 lg4ff_set_range_dfp
  `git blame v6.12 drivers/hid/hid-lg4ff.c`, https://github.com/torvalds/linux.git
*/
static bool HIDAPI_DriverLg4ff_SetRange(SDL_HIDAPI_Device *device, int range)
{
    Uint8 cmd[7] = {0};
    int ret = 0;
    SDL_DriverLg4ff_Context *ctx = (SDL_DriverLg4ff_Context *)device->context;

    if (range < 40) {
        range = 40;
    }
    if (range > 900) {
        range = 900;
    }

    ctx->range = (Uint16)range;
    switch (device->product_id) {
        case USB_DEVICE_ID_LOGITECH_G29_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G27_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G25_WHEEL:
        case USB_DEVICE_ID_LOGITECH_DFGT_WHEEL:{
            cmd[0] = 0xf8;
            cmd[1] = 0x81;
            cmd[2] = range & 0x00ff;
            cmd[3] = (range & 0xff00) >> 8;
            ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
            if (ret == -1) {
                return false;
            }
            break;
        }
        case USB_DEVICE_ID_LOGITECH_DFP_WHEEL:{
            int start_left, start_right, full_range;

            /* Prepare "coarse" limit command */
            cmd[0] = 0xf8;
            cmd[1] = 0x00;    /* Set later */
            cmd[2] = 0x00;
            cmd[3] = 0x00;
            cmd[4] = 0x00;
            cmd[5] = 0x00;
            cmd[6] = 0x00;

            if (range > 200) {
                cmd[1] = 0x03;
                full_range = 900;
            } else {
                cmd[1] = 0x02;
                full_range = 200;
            }
            ret = SDL_hid_write(device->dev, cmd, 7);
            if(ret == -1){
                return false;
            }

            /* Prepare "fine" limit command */
            cmd[0] = 0x81;
            cmd[1] = 0x0b;
            cmd[2] = 0x00;
            cmd[3] = 0x00;
            cmd[4] = 0x00;
            cmd[5] = 0x00;
            cmd[6] = 0x00;

            if (range != 200 && range != 900) {
                /* Construct fine limit command */
                start_left = (((full_range - range + 1) * 2047) / full_range);
                start_right = 0xfff - start_left;

                cmd[2] = (Uint8)(start_left >> 4);
                cmd[3] = (Uint8)(start_right >> 4);
                cmd[4] = 0xff;
                cmd[5] = (start_right & 0xe) << 4 | (start_left & 0xe);
                cmd[6] = 0xff;
            }

            ret = SDL_hid_write(device->dev, cmd, 7);
            if (ret == -1) {
                return false;
            }
            break;
        }
        case USB_DEVICE_ID_LOGITECH_WHEEL:
            // no range setting for ffex/dfex
            break;
        default:
            SDL_assert(0);
    }

    return true;
}

/*
  *Ported*
  Original functions by:
  Simon Wood <simon@mungewell.org>
  Michal Malý <madcatxster@devoid-pointer.net> <madcatxster@gmail.com>
  lg4ff_set_autocenter_default lg4ff_set_autocenter_ffex
  `git blame v6.12 drivers/hid/hid-lg4ff.c`, https://github.com/torvalds/linux.git
*/
static bool HIDAPI_DriverLg4ff_SetAutoCenter(SDL_HIDAPI_Device *device, int magnitude)
{
    SDL_DriverLg4ff_Context *ctx = (SDL_DriverLg4ff_Context *)device->context;
    Uint8 cmd[7] = {0};
    int ret;

    if (magnitude < 0) {
        magnitude = 0;
    }
    if (magnitude > 65535) {
        magnitude = 65535;
    }

    if (ctx->is_ffex) {
        magnitude = magnitude * 90 / 65535;

        cmd[0] = 0xfe;
        cmd[1] = 0x03;
        cmd[2] = (Uint8)((Uint16)magnitude >> 14);
        cmd[3] = (Uint8)((Uint16)magnitude >> 14);
        cmd[4] = (Uint8)magnitude;

        ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
        if(ret == -1){
            return false;
        }
    } else {
        Uint32 expand_a;
        Uint32 expand_b;
        // first disable
        cmd[0] = 0xf5;

        ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
        if (ret == -1) {
            return false;
        }

        if (magnitude == 0) {
            return true;
        }

        // set strength

        if (magnitude <= 0xaaaa) {
            expand_a = 0x0c * magnitude;
            expand_b = 0x80 * magnitude;
        } else {
            expand_a = (0x0c * 0xaaaa) + 0x06 * (magnitude - 0xaaaa);
            expand_b = (0x80 * 0xaaaa) + 0xff * (magnitude - 0xaaaa);
        }
        // TODO do not adjust for MOMO wheels, when support is added
        expand_a = expand_a >> 1;

        SDL_memset(cmd, 0x00, sizeof(cmd));
        cmd[0] = 0xfe;
        cmd[1] = 0x0d;
        cmd[2] = (Uint8)(expand_a / 0xaaaa);
        cmd[3] = (Uint8)(expand_a / 0xaaaa);
        cmd[4] = (Uint8)(expand_b / 0xaaaa);

        ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
        if (ret == -1) {
            return false;
        }

        // enable
        SDL_memset(cmd, 0x00, sizeof(cmd));
        cmd[0] = 0x14;

        ret = SDL_hid_write(device->dev, cmd, sizeof(cmd));
        if (ret == -1) {
            return false;
        }
    }
    return true;
}

/*
  ffex identification method by:
  Simon Wood <simon@mungewell.org>
  Michal Malý <madcatxster@devoid-pointer.net> <madcatxster@gmail.com>
  lg4ff_init
  `git blame v6.12 drivers/hid/hid-lg4ff.c`, https://github.com/torvalds/linux.git
*/
static bool HIDAPI_DriverLg4ff_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverLg4ff_Context *ctx;

    ctx = (SDL_DriverLg4ff_Context *)SDL_malloc(sizeof(SDL_DriverLg4ff_Context));
    if (ctx == NULL) {
        SDL_OutOfMemory();
        return false;
    }
    SDL_memset(ctx, 0, sizeof(SDL_DriverLg4ff_Context));

    device->context = ctx;
    device->joystick_type = SDL_JOYSTICK_TYPE_WHEEL;

    HIDAPI_SetDeviceName(device, HIDAPI_DriverLg4ff_GetDeviceName(device->product_id));

    if (SDL_hid_set_nonblocking(device->dev, 1) != 0) {
        return false;
    }

    if (!HIDAPI_DriverLg4ff_SetAutoCenter(device, 0)) {
        return false;
    }

    if (device->product_id == USB_DEVICE_ID_LOGITECH_WHEEL &&
            (device->version >> 8) == 0x21 &&
            (device->version & 0xff) == 0x00) {
        ctx->is_ffex = true;
    } else {
        ctx->is_ffex = false;
    }

    ctx->range = 900;

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverLg4ff_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverLg4ff_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}


static bool HIDAPI_DriverLg4ff_GetBit(const Uint8 *buf, int bit_num, size_t buf_len)
{
    int byte_offset = bit_num / 8;
    int local_bit = bit_num % 8;
    Uint8 mask = 1 << local_bit;
    if ((size_t)byte_offset >= buf_len) {
        SDL_assert(0);
    }
    return (buf[byte_offset] & mask) ? true : false;
}

/*
  *Ported*
  Original functions by:
  Michal Malý <madcatxster@devoid-pointer.net> <madcatxster@gmail.com>
  lg4ff_adjust_dfp_x_axis
  `git blame v6.12 drivers/hid/hid-lg4ff.c`, https://github.com/torvalds/linux.git
*/
static Uint16 lg4ff_adjust_dfp_x_axis(Uint16 value, Uint16 range)
{
    Uint16 max_range;
    Sint32 new_value;

    if (range == 900)
        return value;
    else if (range == 200)
        return value;
    else if (range < 200)
        max_range = 200;
    else
        max_range = 900;

    new_value = 8192 + ((value - 8192) * max_range / range);
    if (new_value < 0)
        return 0;
    else if (new_value > 16383)
        return 16383;
    else
        return (Uint16)new_value;
}

static bool HIDAPI_DriverLg4ff_HandleState(SDL_HIDAPI_Device *device,
                                               SDL_Joystick *joystick,
                                               Uint8 *report_buf,
                                               size_t report_size)
{
    SDL_DriverLg4ff_Context *ctx = (SDL_DriverLg4ff_Context *)device->context;
    Uint8 hat = 0;
    Uint8 last_hat = 0;
    int num_buttons = HIDAPI_DriverLg4ff_GetNumberOfButtons(device->product_id);
    int bit_offset = 0;
	Uint64 timestamp = SDL_GetTicksNS();

    bool state_changed = false;

    switch (device->product_id) {
        case USB_DEVICE_ID_LOGITECH_G29_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G27_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G25_WHEEL:
        case USB_DEVICE_ID_LOGITECH_DFGT_WHEEL:
            hat = report_buf[0] & 0x0f;
            last_hat = ctx->last_report_buf[0] & 0x0f;
            break;
        case USB_DEVICE_ID_LOGITECH_DFP_WHEEL:
            hat = report_buf[3] >> 4;
            last_hat = ctx->last_report_buf[3] >> 4;
            break;
        case USB_DEVICE_ID_LOGITECH_WHEEL:
            hat = report_buf[2] & 0x0F;
            last_hat = ctx->last_report_buf[2] & 0x0F;
            break;
        default:
            SDL_assert(0);
    }

    if (hat != last_hat) {
        Uint8 sdl_hat = 0;
        state_changed = true;
        switch (hat) {
            case 0:
                sdl_hat = SDL_HAT_UP;
                break;
            case 1:
                sdl_hat = SDL_HAT_RIGHTUP;
                break;
            case 2:
                sdl_hat = SDL_HAT_RIGHT;
                break;
            case 3:
                sdl_hat = SDL_HAT_RIGHTDOWN;
                break;
            case 4:
                sdl_hat = SDL_HAT_DOWN;
                break;
            case 5:
                sdl_hat = SDL_HAT_LEFTDOWN;
                break;
            case 6:
                sdl_hat = SDL_HAT_LEFT;
                break;
            case 7:
                sdl_hat = SDL_HAT_LEFTUP;
                break;
            case 8:
                sdl_hat = SDL_HAT_CENTERED;
                break;
            // do not assert out, in case hardware can report weird hat values
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, sdl_hat);
    }

    switch (device->product_id) {
        case USB_DEVICE_ID_LOGITECH_G29_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G27_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G25_WHEEL:
        case USB_DEVICE_ID_LOGITECH_DFGT_WHEEL:
            bit_offset = 4;
            break;
        case USB_DEVICE_ID_LOGITECH_DFP_WHEEL:
            bit_offset = 14;
            break;
        case USB_DEVICE_ID_LOGITECH_WHEEL:
            bit_offset = 0;
            break;
        default:
            SDL_assert(0);
    }

    for (int i = 0;i < num_buttons;i++) {
        int bit_num = bit_offset + i;
        bool button_on = HIDAPI_DriverLg4ff_GetBit(report_buf, bit_num, report_size);
        bool button_was_on = HIDAPI_DriverLg4ff_GetBit(ctx->last_report_buf, bit_num, report_size);
        if(button_on != button_was_on){
            state_changed = true;
            SDL_SendJoystickButton(timestamp, joystick, (Uint8)(SDL_GAMEPAD_BUTTON_SOUTH + i), button_on);
        }
    }

    switch (device->product_id) {
        case USB_DEVICE_ID_LOGITECH_G29_WHEEL:{
            Uint16 x = *(Uint16 *)&report_buf[4];
            Uint16 last_x = *(Uint16 *)&ctx->last_report_buf[4];
            if (x != last_x) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, x - 32768);
            }
            if (report_buf[6] != ctx->last_report_buf[6]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, report_buf[6] * 257 - 32768);
            }
            if (report_buf[7] != ctx->last_report_buf[7]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, report_buf[7] * 257 - 32768);
            }
            if (report_buf[8] != ctx->last_report_buf[8]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, report_buf[8] * 257 - 32768);
            }
            break;
        }
        case USB_DEVICE_ID_LOGITECH_G27_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G25_WHEEL:{
            Uint16 x = report_buf[4] << 6;
            Uint16 last_x = ctx->last_report_buf[4] << 6;
            x = x | report_buf[3] >> 2;
            last_x = last_x | ctx->last_report_buf[3] >> 2;
            if (x != last_x) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, x * 4 - 32768);
            }
            if (report_buf[5] != ctx->last_report_buf[5]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, report_buf[5] * 257 - 32768);
            }
            if (report_buf[6] != ctx->last_report_buf[6]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, report_buf[6] * 257 - 32768);
            }
            if (report_buf[7] != ctx->last_report_buf[7]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, report_buf[7] * 257 - 32768);
            }
            break;
        }
        case USB_DEVICE_ID_LOGITECH_DFGT_WHEEL:{
            Uint16 x = report_buf[4];
            Uint16 last_x = ctx->last_report_buf[4];
            x = x | (report_buf[5] & 0x3F) << 8;
            last_x = last_x | (ctx->last_report_buf[5] & 0x3F) << 8;
            if (x != last_x) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, x * 4 - 32768);
            }
            if (report_buf[6] != ctx->last_report_buf[6]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, report_buf[6] * 257 - 32768);
            }
            if (report_buf[7] != ctx->last_report_buf[7]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, report_buf[7] * 257 - 32768);
            }
            break;
        }
        case USB_DEVICE_ID_LOGITECH_DFP_WHEEL:{
            Uint16 x = report_buf[0];
            Uint16 last_x = ctx->last_report_buf[0];
            x = x | (report_buf[1] & 0x3F) << 8;
            last_x = last_x | (ctx->last_report_buf[1] & 0x3F) << 8;
            if (x != last_x) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, lg4ff_adjust_dfp_x_axis(x, ctx->range) * 4 - 32768);
            }
            if (report_buf[5] != ctx->last_report_buf[5]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, report_buf[5] * 257 - 32768);
            }
            if (report_buf[6] != ctx->last_report_buf[6]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, report_buf[6] * 257 - 32768);
            }
            break;
        }
        case USB_DEVICE_ID_LOGITECH_WHEEL:{
            if (report_buf[3] != ctx->last_report_buf[3]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, report_buf[3] * 257 - 32768);
            }
            if (report_buf[4] != ctx->last_report_buf[4]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, report_buf[4] * 257 - 32768);
            }
            if (report_buf[5] != ctx->last_report_buf[5]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, report_buf[5] * 257 - 32768);
            }
            if (report_buf[6] != ctx->last_report_buf[6]) {
                state_changed = true;
                SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, report_buf[7] * 257 - 32768);
            }
            break;
        }
        default:
            SDL_assert(0);
    }

    SDL_memcpy(ctx->last_report_buf, report_buf, report_size);
    return state_changed;
}

static bool HIDAPI_DriverLg4ff_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_Joystick *joystick = NULL;
    int r;
    Uint8 report_buf[32] = {0};
    size_t report_size = 0;
    SDL_DriverLg4ff_Context *ctx = (SDL_DriverLg4ff_Context *)device->context;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
        if (joystick == NULL) {
            return false;
        }
    } else {
        return false;
    }

    switch (device->product_id) {
        case USB_DEVICE_ID_LOGITECH_G29_WHEEL:
            report_size = 12;
            break;
        case USB_DEVICE_ID_LOGITECH_G27_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G25_WHEEL:
            report_size = 11;
            break;
        case USB_DEVICE_ID_LOGITECH_DFGT_WHEEL:
        case USB_DEVICE_ID_LOGITECH_DFP_WHEEL:
            report_size = 8;
            break;
        case USB_DEVICE_ID_LOGITECH_WHEEL:
            report_size = 27;
            break;
        default:
            SDL_assert(0);
    }

    do {
        r = SDL_hid_read(device->dev, report_buf, report_size);
        if (r < 0) {
            /* Failed to read from controller */
            HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
            return false;
        } else if ((size_t)r == report_size) {
            bool state_changed = HIDAPI_DriverLg4ff_HandleState(device, joystick, report_buf, report_size);
            if(state_changed && !ctx->initialized) {
                ctx->initialized = true;
                HIDAPI_DriverLg4ff_SetRange(device, SDL_HIDAPI_DriverLg4ff_GetEnvInt("SDL_HIDAPI_LG4FF_RANGE", 40, 900, 900));
                HIDAPI_DriverLg4ff_SetAutoCenter(device, 0);
            }
        }
    } while (r > 0);

    return true;
}

static bool HIDAPI_DriverLg4ff_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_AssertJoysticksLocked();

    // Initialize the joystick capabilities
    joystick->nhats = 1;
    joystick->nbuttons = HIDAPI_DriverLg4ff_GetNumberOfButtons(device->product_id);
    switch(device->product_id){
        case USB_DEVICE_ID_LOGITECH_G29_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G27_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G25_WHEEL:
        case USB_DEVICE_ID_LOGITECH_WHEEL:
            joystick->naxes = 4;
            break;
        case USB_DEVICE_ID_LOGITECH_DFGT_WHEEL:
            joystick->naxes = 3;
            break;
        case USB_DEVICE_ID_LOGITECH_DFP_WHEEL:
            joystick->naxes = 3;
            break;
        default:
            SDL_assert(0);
    }

    return true;
}

static bool HIDAPI_DriverLg4ff_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverLg4ff_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverLg4ff_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    switch(device->product_id) {
        case USB_DEVICE_ID_LOGITECH_G29_WHEEL:
        case USB_DEVICE_ID_LOGITECH_G27_WHEEL:
            return SDL_JOYSTICK_CAP_MONO_LED;
        default:
            return 0;
    }
}

/*
  Commands by:
  Michal Malý <madcatxster@devoid-pointer.net> <madcatxster@gmail.com>
  Simon Wood <simon@mungewell.org>
  lg4ff_led_set_brightness lg4ff_set_leds
  `git blame v6.12 drivers/hid/hid-lg4ff.c`, https://github.com/torvalds/linux.git
*/
static bool HIDAPI_DriverLg4ff_SendLedCommand(SDL_HIDAPI_Device *device, Uint8 state)
{
    Uint8 cmd[7];
    Uint8 led_state = 0;

    switch (state) {
        case 0:
            led_state = 0;
            break;
        case 1:
            led_state = 1;
            break;
        case 2:
            led_state = 3;
            break;
        case 3:
            led_state = 7;
            break;
        case 4:
            led_state = 15;
            break;
        case 5:
            led_state = 31;
            break;
        default:
            SDL_assert(0);
    }

    cmd[0] = 0xf8;
    cmd[1] = 0x12;
    cmd[2] = led_state;
    cmd[3] = 0x00;
    cmd[4] = 0x00;
    cmd[5] = 0x00;
    cmd[6] = 0x00;

    return SDL_hid_write(device->dev, cmd, sizeof(cmd)) == sizeof(cmd);
}

static bool HIDAPI_DriverLg4ff_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    int max_led = red;

    // only g27/g29, and g923 when supported is added
    if (device->product_id != USB_DEVICE_ID_LOGITECH_G29_WHEEL &&
    device->product_id != USB_DEVICE_ID_LOGITECH_G27_WHEEL) {
        return SDL_Unsupported();
    }

    if (green > max_led) {
        max_led = green;
    }
    if (blue > max_led) {
        max_led = blue;
    }

    return HIDAPI_DriverLg4ff_SendLedCommand(device, (Uint8)((5 * max_led) / 255));
}

static bool HIDAPI_DriverLg4ff_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    // allow programs to send raw commands
    return SDL_hid_write(device->dev, data, size) == size;
}

static bool HIDAPI_DriverLg4ff_SetSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    // On steam deck, sensors are enabled by default. Nothing to do here.
    return SDL_Unsupported();
}

static void HIDAPI_DriverLg4ff_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    // remember to stop effects on haptics close, when implemented
    HIDAPI_DriverLg4ff_SetJoystickLED(device, joystick, 0, 0, 0);
}

static void HIDAPI_DriverLg4ff_FreeDevice(SDL_HIDAPI_Device *device)
{
    // device context is freed in SDL_hidapijoystick.c
}


SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverLg4ff = {
    SDL_HINT_JOYSTICK_HIDAPI_LG4FF,
    true,
    HIDAPI_DriverLg4ff_RegisterHints,
    HIDAPI_DriverLg4ff_UnregisterHints,
    HIDAPI_DriverLg4ff_IsEnabled,
    HIDAPI_DriverLg4ff_IsSupportedDevice,
    HIDAPI_DriverLg4ff_InitDevice,
    HIDAPI_DriverLg4ff_GetDevicePlayerIndex,
    HIDAPI_DriverLg4ff_SetDevicePlayerIndex,
    HIDAPI_DriverLg4ff_UpdateDevice,
    HIDAPI_DriverLg4ff_OpenJoystick,
    HIDAPI_DriverLg4ff_RumbleJoystick,
    HIDAPI_DriverLg4ff_RumbleJoystickTriggers,
    HIDAPI_DriverLg4ff_GetJoystickCapabilities,
    HIDAPI_DriverLg4ff_SetJoystickLED,
    HIDAPI_DriverLg4ff_SendJoystickEffect,
    HIDAPI_DriverLg4ff_SetSensorsEnabled,
    HIDAPI_DriverLg4ff_CloseJoystick,
    HIDAPI_DriverLg4ff_FreeDevice,
};


#endif /* SDL_JOYSTICK_HIDAPI_LG4FF */

#endif /* SDL_JOYSTICK_HIDAPI */
