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
/* This driver supports the Nintendo Switch Pro controller.
   Code and logic contributed by Valve Corporation under the SDL zlib license.
*/
#include "SDL_internal.h"

#ifdef SDL_JOYSTICK_HIDAPI

#include "../../SDL_hints_c.h"
#include "../../misc/SDL_libusb.h"
#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"

#ifdef SDL_JOYSTICK_HIDAPI_SWITCH2

#define RUMBLE_INTERVAL 12
#define RUMBLE_MAX 29000

// Define this if you want to log all packets from the controller
#if 0
#define DEBUG_SWITCH2_PROTOCOL
#endif

enum
{
    SDL_GAMEPAD_BUTTON_SWITCH2_PRO_SHARE = 11,
    SDL_GAMEPAD_BUTTON_SWITCH2_PRO_C,
    SDL_GAMEPAD_BUTTON_SWITCH2_PRO_RIGHT_PADDLE,
    SDL_GAMEPAD_BUTTON_SWITCH2_PRO_LEFT_PADDLE,
    SDL_GAMEPAD_NUM_SWITCH2_PRO_BUTTONS
};

enum
{
    SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_SHARE = 11,
    SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_C,
    SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_RIGHT_PADDLE1,
    SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_LEFT_PADDLE1,
    SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_RIGHT_PADDLE2,
    SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_LEFT_PADDLE2,
    SDL_GAMEPAD_NUM_SWITCH2_JOYCON_BUTTONS
};

enum
{
    SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_GUIDE = 4,
    SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_START,
    SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_LEFT_SHOULDER,
    SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_RIGHT_SHOULDER,
    SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_SHARE,
    SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_C,
    SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_LEFT_TRIGGER,   // Full trigger pull click
    SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_RIGHT_TRIGGER,  // Full trigger pull click
    SDL_GAMEPAD_NUM_SWITCH2_GAMECUBE_BUTTONS
};

typedef struct
{
    Uint16 neutral;
    Uint16 max;
    Uint16 min;
} Switch2_AxisCalibration;

typedef struct
{
    Switch2_AxisCalibration x;
    Switch2_AxisCalibration y;
} Switch2_StickCalibration;

typedef struct
{
    SDL_HIDAPI_Device *device;
    SDL_Joystick *joystick;

    SDL_LibUSBContext *libusb;
    libusb_device_handle *device_handle;
    bool interface_claimed;
    Uint8 interface_number;
    Uint8 out_endpoint;
    Uint8 in_endpoint;

    Uint64 rumble_timestamp;
    Uint32 rumble_seq;
    Uint16 rumble_hi_amp;
    Uint16 rumble_hi_freq;
    Uint16 rumble_lo_amp;
    Uint16 rumble_lo_freq;
    Uint32 rumble_error;
    bool rumble_updated;

    Switch2_StickCalibration left_stick;
    Switch2_StickCalibration right_stick;
    Uint8 left_trigger_zero;
    Uint8 right_trigger_zero;

    float gyro_bias_x;
    float gyro_bias_y;
    float gyro_bias_z;
    float accel_bias_x;
    float accel_bias_y;
    float accel_bias_z;
    bool sensors_enabled;
    bool sensors_ready;
    int sample_count;
    Uint64 first_sensor_timestamp;
    Uint64 sensor_ts_coeff;
    float gyro_coeff;

    bool player_lights;
    int player_index;

    bool vertical_mode;
    Uint8 last_state[USB_PACKET_LENGTH];
} SDL_DriverSwitch2_Context;

static void ParseStickCalibration(Switch2_StickCalibration *stick_data, const Uint8 *data)
{
    stick_data->x.neutral = data[0];
    stick_data->x.neutral |= (data[1] & 0x0F) << 8;

    stick_data->y.neutral = data[1] >> 4;
    stick_data->y.neutral |= data[2] << 4;

    stick_data->x.max = data[3];
    stick_data->x.max |= (data[4] & 0x0F) << 8;

    stick_data->y.max = data[4] >> 4;
    stick_data->y.max |= data[5] << 4;

    stick_data->x.min = data[6];
    stick_data->x.min |= (data[7] & 0x0F) << 8;

    stick_data->y.min = data[7] >> 4;
    stick_data->y.min |= data[8] << 4;
}

static int SendBulkData(SDL_DriverSwitch2_Context *ctx, const Uint8 *data, unsigned size)
{
    int transferred;
    int res = ctx->libusb->bulk_transfer(ctx->device_handle,
                ctx->out_endpoint,
                (Uint8 *)data,
                size,
                &transferred,
                1000);
    if (res < 0) {
        return res;
    }
    return transferred;
}

static int RecvBulkData(SDL_DriverSwitch2_Context *ctx, Uint8 *data, unsigned size)
{
    int transferred;
    int total_transferred = 0;
    int res;

    while (size > 0) {
        unsigned current_read = size;
        if (current_read > 64) {
            current_read = 64;
        }
        res = ctx->libusb->bulk_transfer(ctx->device_handle,
                    ctx->in_endpoint,
                    data,
                    current_read,
                    &transferred,
                    100);
        if (res < 0) {
            return res;
        }
        total_transferred += transferred;
        size -= transferred;
        data += current_read;
        if ((unsigned) transferred < current_read) {
            break;
        }
    }

    return total_transferred;
}

static void MapJoystickAxis(Uint64 timestamp, SDL_Joystick *joystick, Uint8 axis, const Switch2_AxisCalibration *calib, float value, bool invert)
{
    Sint16 mapped_value;
    if (calib && calib->neutral && calib->min && calib->max) {
        value -= calib->neutral;
        if (value < 0) {
            value /= calib->min;
        } else {
            value /= calib->max;
        }
        mapped_value = (Sint16) SDL_clamp(value * SDL_MAX_SINT16, SDL_MIN_SINT16, SDL_MAX_SINT16);
    } else {
        mapped_value = (Sint16) HIDAPI_RemapVal(value, 0, 4096, SDL_MIN_SINT16, SDL_MAX_SINT16);
    }
    if (invert) {
        mapped_value = ~mapped_value;
    }
    SDL_SendJoystickAxis(timestamp, joystick, axis, mapped_value);
}

static void MapTriggerAxis(Uint64 timestamp, SDL_Joystick *joystick, Uint8 axis, Uint8 max, float value)
{
    Sint16 mapped_value = (Sint16) HIDAPI_RemapVal(
        SDL_clamp((value - max) / (232.f - max), 0, 1),
        0, 1,
        SDL_MIN_SINT16, SDL_MAX_SINT16
    );
    SDL_SendJoystickAxis(timestamp, joystick, axis, mapped_value);
}

static bool UpdateSlotLED(SDL_DriverSwitch2_Context *ctx)
{
    Uint8 set_led_data[] = {
        0x09, 0x91, 0x00, 0x07, 0x00, 0x08, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    };
    Uint8 reply[8] = {0};
    const Uint8 player_pattern[] = { 0x1, 0x3, 0x7, 0xf, 0x9, 0x5, 0xd, 0x6 };

    if (ctx->player_lights && ctx->player_index >= 0) {
        set_led_data[8] = player_pattern[ctx->player_index % 8];
    }
    int res = SendBulkData(ctx, set_led_data, sizeof(set_led_data));
    if (res < 0) {
        return SDL_SetError("Couldn't set LED data: %d\n", res);
    }
    return (RecvBulkData(ctx, reply, sizeof(reply)) > 0);
}

static int ReadFlashBlock(SDL_DriverSwitch2_Context *ctx, Uint32 address, Uint8 *out)
{
    Uint8 flash_read_command[] = {
        0x02, 0x91, 0x00, 0x01, 0x00, 0x08, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    };
    Uint8 buffer[0x50] = {0};
    int res;

    flash_read_command[12] = (Uint8)address;
    flash_read_command[13] = (Uint8)(address >> 8);
    flash_read_command[14] = (Uint8)(address >> 16);
    flash_read_command[15] = (Uint8)(address >> 24);

    res = SendBulkData(ctx, flash_read_command, sizeof(flash_read_command));
    if (res < 0) {
        return res;
    }

    res = RecvBulkData(ctx, buffer, sizeof(buffer));
    if (res < 0) {
        return res;
    }

    SDL_memcpy(out, &buffer[0x10], 0x40);
    return 0;
}

static void SDLCALL SDL_PlayerLEDHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_DriverSwitch2_Context *ctx = (SDL_DriverSwitch2_Context *)userdata;
    bool player_lights = SDL_GetStringBoolean(hint, true);

    if (player_lights != ctx->player_lights) {
        ctx->player_lights = player_lights;

        UpdateSlotLED(ctx);
        HIDAPI_UpdateDeviceProperties(ctx->device);
    }
}

static void HIDAPI_DriverSwitch2_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SWITCH2, callback, userdata);
}

static void HIDAPI_DriverSwitch2_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SWITCH2, callback, userdata);
}

static bool HIDAPI_DriverSwitch2_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_SWITCH2, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverSwitch2_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (vendor_id == USB_VENDOR_NINTENDO) {
        switch (product_id) {
        case USB_PRODUCT_NINTENDO_SWITCH2_GAMECUBE_CONTROLLER:
        case USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_LEFT:
        case USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_RIGHT:
        case USB_PRODUCT_NINTENDO_SWITCH2_PRO:
            return true;
        }
    }

    return false;
}

static bool HIDAPI_DriverSwitch2_InitBluetooth(SDL_HIDAPI_Device *device)
{
    // FIXME: Need to add Bluetooth support
    return SDL_SetError("Nintendo Switch2 controllers not supported over Bluetooth");
}

static bool FindBulkEndpoints(SDL_LibUSBContext *libusb, libusb_device_handle *handle, Uint8 *bInterfaceNumber, Uint8 *out_endpoint, Uint8 *in_endpoint)
{
    struct libusb_config_descriptor *config;
    int found = 0;

    if (libusb->get_config_descriptor(libusb->get_device(handle), 0, &config) != 0) {
         return false;
    }

    for (int i = 0; i < config->bNumInterfaces; i++) {
        const struct libusb_interface *iface = &config->interface[i];
        for (int j = 0; j < iface->num_altsetting; j++) {
            const struct libusb_interface_descriptor *altsetting = &iface->altsetting[j];
            if (altsetting->bInterfaceNumber == 1) {
                for (int k = 0; k < altsetting->bNumEndpoints; k++) {
                    const struct libusb_endpoint_descriptor *ep = &altsetting->endpoint[k];
                    if ((ep->bmAttributes & LIBUSB_TRANSFER_TYPE_MASK) == LIBUSB_TRANSFER_TYPE_BULK) {
                        *bInterfaceNumber = altsetting->bInterfaceNumber;
                        if ((ep->bEndpointAddress & LIBUSB_ENDPOINT_DIR_MASK) == LIBUSB_ENDPOINT_OUT) {
                            *out_endpoint = ep->bEndpointAddress;
                            found |= 1;
                        }
                        if ((ep->bEndpointAddress & LIBUSB_ENDPOINT_DIR_MASK) == LIBUSB_ENDPOINT_IN) {
                            *in_endpoint = ep->bEndpointAddress;
                            found |= 2;
                        }
                        if (found == 3) {
                            libusb->free_config_descriptor(config);
                            return true;
                        }
                    }
                }
            }
        }
    }
    libusb->free_config_descriptor(config);
    return false;
}

static bool HIDAPI_DriverSwitch2_InitUSB(SDL_HIDAPI_Device *device)
{
    SDL_DriverSwitch2_Context *ctx = (SDL_DriverSwitch2_Context *)device->context;

    if (!SDL_InitLibUSB(&ctx->libusb)) {
        return false;
    }

    ctx->device_handle = (libusb_device_handle *)SDL_GetPointerProperty(SDL_hid_get_properties(device->dev), SDL_PROP_HIDAPI_LIBUSB_DEVICE_HANDLE_POINTER, NULL);
    if (!ctx->device_handle) {
        return SDL_SetError("Couldn't get libusb device handle");
    }

    if (!FindBulkEndpoints(ctx->libusb, ctx->device_handle, &ctx->interface_number, &ctx->out_endpoint, &ctx->in_endpoint)) {
        return SDL_SetError("Couldn't find bulk endpoints");
    }

    ctx->libusb->set_auto_detach_kernel_driver(ctx->device_handle, true);
    int res = ctx->libusb->claim_interface(ctx->device_handle, ctx->interface_number);
    if (res < 0) {
        return SDL_SetError("Couldn't claim interface %d: %d\n", ctx->interface_number, res);
    }
    ctx->interface_claimed = true;

    const Uint8 *init_sequence[] = {
        (Uint8[]) { // Unknown purpose
            0x7, 0x91, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0,
        },
        (Uint8[]) { // Set feature output bit mask
            0x0c, 0x91, 0x00, 0x02, 0x00, 0x04, 0x00, 0x00, 0x27, 0x00, 0x00, 0x00
        },
        (Uint8[]) { // Unknown purpose
            0x11, 0x91, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0,
        },
        (Uint8[]) { // Set rumble data?
            0x0a, 0x91, 0x00, 0x08, 0x00, 0x14, 0x00, 0x00,
            0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0x35, 0x00, 0x46, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00
        },
        (Uint8[]) { // Enable feature output bits
            0x0c, 0x91, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x27, 0x00, 0x00, 0x00
        },
        (Uint8[]) { // Unknown purpose
            0x01, 0x91, 0x0, 0xc, 0x0, 0x0, 0x0, 0x0,
        },
        (Uint8[]) { // Enable rumble
            0x01, 0x91, 0x0, 0x1, 0x0, 0x0, 0x0, 0x0,
        },
        (Uint8[]) { // Enable grip buttons on charging grip
            0x8, 0x91, 0x0, 0x2, 0x0, 0x4, 0x0, 0x0, 0x01, 0x0, 0x0, 0x0,
        },
        (Uint8[]) { // Set report format
            0x03, 0x91, 0x00, 0x0a, 0x00, 0x04, 0x00, 0x00,
            0x05, 0x00, 0x00, 0x00
        },
        (Uint8[]) { // Start output
            0x03, 0x91, 0x00, 0x0d, 0x00, 0x08, 0x00, 0x00,
            0x01, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        },
        NULL, // Sentinel
    };

    unsigned char calibration_data[0x40] = {0};

    res = ReadFlashBlock(ctx, 0x13000, calibration_data);
    if (res < 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't read serial number: %d", res);
    } else {
        char serial[0x11] = {0};
        SDL_strlcpy(serial, (char*)&calibration_data[2], sizeof(serial));
        HIDAPI_SetDeviceSerial(device, serial);
    }

    res = ReadFlashBlock(ctx, 0x13040, calibration_data);
    if (res < 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't read factory calibration data: %d", res);
    } else {
        ctx->gyro_bias_x = *(float*)&calibration_data[4];
        ctx->gyro_bias_y = *(float*)&calibration_data[8];
        ctx->gyro_bias_z = *(float*)&calibration_data[12];
    }

    res = ReadFlashBlock(ctx, 0x13080, calibration_data);
    if (res < 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't read factory calibration data: %d", res);
    } else {
        ParseStickCalibration(&ctx->left_stick, &calibration_data[0x28]);
    }

    res = ReadFlashBlock(ctx, 0x130C0, calibration_data);
    if (res < 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't read factory calibration data: %d", res);
    } else {
        ParseStickCalibration(&ctx->right_stick, &calibration_data[0x28]);
    }

    res = ReadFlashBlock(ctx, 0x13100, calibration_data);
    if (res < 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't read factory calibration data: %d", res);
    } else {
        ctx->accel_bias_x = *(float*)&calibration_data[12];
        ctx->accel_bias_y = *(float*)&calibration_data[16];
        ctx->accel_bias_z = *(float*)&calibration_data[20];
    }

    if (device->product_id == USB_PRODUCT_NINTENDO_SWITCH2_GAMECUBE_CONTROLLER) {
        res = ReadFlashBlock(ctx, 0x13140, calibration_data);
        if (res < 0) {
            SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't read factory calibration data: %d", res);
        } else {
            ctx->left_trigger_zero = calibration_data[0];
            ctx->right_trigger_zero = calibration_data[1];
        }
    }

    res = ReadFlashBlock(ctx, 0x1FC040, calibration_data);
    if (res < 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't read user calibration data: %d", res);
    } else if (calibration_data[0] == 0xb2 && calibration_data[1] == 0xa1) {
        ParseStickCalibration(&ctx->left_stick, &calibration_data[2]);
    }

    res = ReadFlashBlock(ctx, 0x1FC080, calibration_data);
    if (res < 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "Couldn't read user calibration data: %d", res);
    } else if (calibration_data[0] == 0xb2 && calibration_data[1] == 0xa1) {
        ParseStickCalibration(&ctx->right_stick, &calibration_data[2]);
    }

    for (int i = 0; init_sequence[i]; i++) {
        res = SendBulkData(ctx, init_sequence[i], init_sequence[i][5] + 8);
        if (res < 0) {
            return SDL_SetError("Couldn't send initialization data: %d\n", res);
        }
        RecvBulkData(ctx, calibration_data, 0x40);
    }

    return true;
}

static bool HIDAPI_DriverSwitch2_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSwitch2_Context *ctx;

    ctx = (SDL_DriverSwitch2_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    ctx->device = device;
    device->context = ctx;

    if (device->is_bluetooth) {
        if (!HIDAPI_DriverSwitch2_InitBluetooth(device)) {
            return false;
        }
    } else {
        if (!HIDAPI_DriverSwitch2_InitUSB(device)) {
            return false;
        }
    }

    ctx->sensor_ts_coeff = 10000;
    ctx->gyro_coeff = 34.8f;

    // Sometimes the device handle isn't available during enumeration so we don't get the device name, so set it explicitly
    switch (device->product_id) {
    case USB_PRODUCT_NINTENDO_SWITCH2_GAMECUBE_CONTROLLER:
        HIDAPI_SetDeviceName(device, "Nintendo GameCube Controller");
        break;
    case USB_PRODUCT_NINTENDO_SWITCH2_PRO:
        HIDAPI_SetDeviceName(device, "Nintendo Switch Pro Controller");
        break;
    default:
        break;
    }
    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverSwitch2_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverSwitch2_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
    SDL_DriverSwitch2_Context *ctx = (SDL_DriverSwitch2_Context *)device->context;

    if (!ctx->joystick) {
        return;
    }

    ctx->player_index = player_index;

    UpdateSlotLED(ctx);
}

static bool HIDAPI_DriverSwitch2_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverSwitch2_Context *ctx = (SDL_DriverSwitch2_Context *)device->context;

    ctx->joystick = joystick;

    // Initialize player index (needed for setting LEDs)
    ctx->player_index = SDL_GetJoystickPlayerIndex(joystick);
    ctx->player_lights = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_SWITCH_PLAYER_LED, true);
    UpdateSlotLED(ctx);

    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SWITCH_PLAYER_LED,
                        SDL_PlayerLEDHintChanged, ctx);

    // Initialize the joystick capabilities
    if (!ctx->device->parent) {
        SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO, 250.0f);
        SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, 250.0f);
    }
    switch (device->product_id) {
    case USB_PRODUCT_NINTENDO_SWITCH2_GAMECUBE_CONTROLLER:
        joystick->nbuttons = SDL_GAMEPAD_NUM_SWITCH2_GAMECUBE_BUTTONS;
        break;
    case USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_LEFT:
        if (ctx->device->parent) {
            SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO_L, 250.0f);
            SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL_L, 250.0f);
        }
        joystick->nbuttons = SDL_GAMEPAD_NUM_SWITCH2_JOYCON_BUTTONS;
        break;
    case USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_RIGHT:
        if (ctx->device->parent) {
            SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO, 250.0f);
            SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, 250.0f);
            SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO_R, 250.0f);
            SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL_R, 250.0f);
        }
        joystick->nbuttons = SDL_GAMEPAD_NUM_SWITCH2_JOYCON_BUTTONS;
        break;
    case USB_PRODUCT_NINTENDO_SWITCH2_PRO:
        joystick->nbuttons = SDL_GAMEPAD_NUM_SWITCH2_PRO_BUTTONS;
        break;
    default:
        // FIXME: How many buttons does this have?
        break;
    }
    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
    joystick->nhats = 1;

    ctx->rumble_hi_freq = 0x187;
    ctx->rumble_lo_freq = 0x112;

    // Set up for vertical mode
    ctx->vertical_mode = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_VERTICAL_JOY_CONS, false);

    return true;
}

static bool HIDAPI_DriverSwitch2_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverSwitch2_Context *ctx = (SDL_DriverSwitch2_Context *)device->context;

    if (low_frequency_rumble != ctx->rumble_lo_amp || high_frequency_rumble != ctx->rumble_hi_amp) {
        ctx->rumble_lo_amp = low_frequency_rumble;
        ctx->rumble_hi_amp = high_frequency_rumble;
        ctx->rumble_updated = true;
    }

    return true;
}

static bool HIDAPI_DriverSwitch2_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverSwitch2_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverSwitch2_Context *ctx = (SDL_DriverSwitch2_Context *)device->context;
    Uint32 result = SDL_JOYSTICK_CAP_RUMBLE;

    if (ctx->player_lights) {
        result |= SDL_JOYSTICK_CAP_PLAYER_LED;
    }
    return result;
}

static bool HIDAPI_DriverSwitch2_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSwitch2_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSwitch2_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    SDL_DriverSwitch2_Context *ctx = (SDL_DriverSwitch2_Context *)device->context;
    if (ctx->sensors_ready) {
        Uint8 data[] = {
                0x0c, 0x91, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00
        };
        unsigned char reply[12] = {0};

        if (enabled) {
            data[8] |= 4;
        }
        int res = SendBulkData(ctx, data, sizeof(data));
        if (res < 0) {
            return SDL_SetError("Couldn't set sensors enabled: %d\n", res);
        }
        RecvBulkData(ctx, reply, sizeof(reply));
    }
    ctx->sensors_enabled = true;
    return true;
}

static void HandleGameCubeState(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch2_Context *ctx, Uint8 *data, int size)
{

    if (data[5] != ctx->last_state[5]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[5] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[5] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[5] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[5] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_RIGHT_TRIGGER, ((data[5] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_RIGHT_SHOULDER, ((data[5] & 0x80) != 0));
    }

    if (data[6] != ctx->last_state[6]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_START, ((data[6] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_GUIDE, ((data[6] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_SHARE, ((data[6] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_C, ((data[6] & 0x40) != 0));
    }

    if (data[7] != ctx->last_state[7]) {
        Uint8 hat = 0;

        if (data[7] & 0x01) {
            hat |= SDL_HAT_DOWN;
        }
        if (data[7] & 0x02) {
            hat |= SDL_HAT_UP;
        }
        if (data[7] & 0x04) {
            hat |= SDL_HAT_RIGHT;
        }
        if (data[7] & 0x08) {
            hat |= SDL_HAT_LEFT;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_LEFT_TRIGGER, ((data[7] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_GAMECUBE_LEFT_SHOULDER, ((data[7] & 0x80) != 0));
    }

    MapTriggerAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFT_TRIGGER,
        ctx->left_trigger_zero,
        data[61]
    );
    MapTriggerAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_RIGHT_TRIGGER,
        ctx->right_trigger_zero,
        data[62]
    );

    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFTX,
        &ctx->left_stick.x,
        (float) (data[11] | ((data[12] & 0x0F) << 8)),
        false
    );
    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFTY,
        &ctx->left_stick.y,
        (float) ((data[12] >> 4) | (data[13] << 4)),
        true
    );
    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_RIGHTX,
        &ctx->right_stick.x,
        (float) (data[14] | ((data[15] & 0x0F) << 8)),
        false
    );
    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_RIGHTY,
        &ctx->right_stick.y,
        (float)((data[15] >> 4) | (data[16] << 4)),
        true
    );
}

static void HandleCombinedControllerStateL(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch2_Context *ctx, Uint8 *data, int size)
{
    if (data[6] != ctx->last_state[6]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[6] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[6] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_SHARE, ((data[6] & 0x20) != 0));
    }

    if (data[7] != ctx->last_state[7]) {
        Uint8 hat = 0;

        if (data[7] & 0x01) {
            hat |= SDL_HAT_DOWN;
        }
        if (data[7] & 0x02) {
            hat |= SDL_HAT_UP;
        }
        if (data[7] & 0x04) {
            hat |= SDL_HAT_RIGHT;
        }
        if (data[7] & 0x08) {
            hat |= SDL_HAT_LEFT;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[7] & 0x40) != 0));
    }

    if (data[8] != ctx->last_state[8]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_LEFT_PADDLE1, ((data[8] & 0x02) != 0));
    }

    Sint16 axis = (data[7] & 0x80) ? 32767 : -32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);

    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFTX,
        &ctx->left_stick.x,
        (float) (data[11] | ((data[12] & 0x0F) << 8)),
        false
    );
    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFTY,
        &ctx->left_stick.y,
        (float) ((data[12] >> 4) | (data[13] << 4)),
        true
    );
}

static void HandleMiniControllerStateL(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch2_Context *ctx, Uint8 *data, int size)
{
    if (data[6] != ctx->last_state[6]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[6] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[6] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[6] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_SHARE, ((data[6] & 0x10) != 0));
    }

    if (data[7] != ctx->last_state[7]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[7] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[7] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[7] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[7] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[7] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[7] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_LEFT_PADDLE1, ((data[7] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_LEFT_PADDLE2, ((data[7] & 0x80) != 0));
    }

    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFTX,
        &ctx->left_stick.y,
        (float) ((data[12] >> 4) | (data[13] << 4)),
        true
    );
    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFTY,
        &ctx->left_stick.x,
        (float) (data[11] | ((data[12] & 0x0F) << 8)),
        true
    );
}

static void HandleCombinedControllerStateR(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch2_Context *ctx, Uint8 *data, int size)
{
    if (data[5] != ctx->last_state[5]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[5] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[5] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[5] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[5] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[5] & 0x40) != 0));
    }

    if (data[6] != ctx->last_state[6]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[6] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[6] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[6] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_C, ((data[6] & 0x40) != 0));
    }

    if (data[8] != ctx->last_state[8]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_RIGHT_PADDLE1, ((data[8] & 0x01) != 0));
    }

    Sint16 axis = (data[5] & 0x80) ? 32767 : -32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);

    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_RIGHTX,
        &ctx->left_stick.x,
        (float) (data[14] | ((data[15] & 0x0F) << 8)),
        false
    );
    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_RIGHTY,
        &ctx->left_stick.y,
        (float)((data[15] >> 4) | (data[16] << 4)),
        true
    );
}

static void HandleMiniControllerStateR(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch2_Context *ctx, Uint8 *data, int size)
{
    if (data[5] != ctx->last_state[5]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[5] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[5] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[5] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[5] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[5] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[5] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_RIGHT_PADDLE1, ((data[5] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_RIGHT_PADDLE2, ((data[5] & 0x80) != 0));
    }

    if (data[6] != ctx->last_state[6]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[6] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[6] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[6] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_JOYCON_C, ((data[6] & 0x40) != 0));
    }

    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFTX,
        &ctx->left_stick.y,
        (float)((data[15] >> 4) | (data[16] << 4)),
        false
    );
    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFTY,
        &ctx->left_stick.x,
        (float) (data[14] | ((data[15] & 0x0F) << 8)),
        false
    );
}

static void HandleSwitchProState(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch2_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;

    if (data[5] != ctx->last_state[5]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[5] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[5] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[5] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[5] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[5] & 0x40) != 0));
    }

    if (data[6] != ctx->last_state[6]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[6] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[6] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[6] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[6] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[6] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_PRO_SHARE, ((data[6] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_PRO_C, ((data[6] & 0x40) != 0));
    }

    if (data[7] != ctx->last_state[7]) {
        Uint8 hat = 0;

        if (data[7] & 0x01) {
            hat |= SDL_HAT_DOWN;
        }
        if (data[7] & 0x02) {
            hat |= SDL_HAT_UP;
        }
        if (data[7] & 0x04) {
            hat |= SDL_HAT_RIGHT;
        }
        if (data[7] & 0x08) {
            hat |= SDL_HAT_LEFT;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[7] & 0x40) != 0));
    }

    if (data[8] != ctx->last_state[8]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_PRO_RIGHT_PADDLE, ((data[8] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH2_PRO_LEFT_PADDLE, ((data[8] & 0x02) != 0));
    }

    axis = (data[5] & 0x80) ? 32767 : -32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);

    axis = (data[7] & 0x80) ? 32767 : -32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);

    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFTX,
        &ctx->left_stick.x,
        (float) (data[11] | ((data[12] & 0x0F) << 8)),
        false
    );
    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_LEFTY,
        &ctx->left_stick.y,
        (float) ((data[12] >> 4) | (data[13] << 4)),
        true
    );
    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_RIGHTX,
        &ctx->right_stick.x,
        (float) (data[14] | ((data[15] & 0x0F) << 8)),
        false
    );
    MapJoystickAxis(
        timestamp,
        joystick,
        SDL_GAMEPAD_AXIS_RIGHTY,
        &ctx->right_stick.y,
        (float)((data[15] >> 4) | (data[16] << 4)),
        true
    );
}

static void EncodeHDRumble(Uint16 high_freq, Uint16 high_amp, Uint16 low_freq, Uint16 low_amp, Uint8 rumble_data[5])
{
    rumble_data[0] = (Uint8)(high_freq & 0xFF);
    rumble_data[1] = (Uint8)(((high_amp >> 4) & 0xfc) | ((high_freq >> 8) & 0x03));
    rumble_data[2] = (Uint8)((high_amp >> 12) | (low_freq << 4));
    rumble_data[3] = (Uint8)((low_amp & 0xc0) | ((low_freq >> 4) & 0x3f));
    rumble_data[4] = (Uint8)(low_amp >> 8);
}

static bool UpdateRumble(SDL_DriverSwitch2_Context *ctx)
{
    if (!ctx->rumble_updated && !ctx->rumble_lo_amp && !ctx->rumble_hi_amp) {
        return true;
    }

    Uint64 timestamp = SDL_GetTicks();
    Uint64 interval = RUMBLE_INTERVAL;

    if (timestamp < ctx->rumble_timestamp) {
        return true;
    }

    if (!SDL_HIDAPI_LockRumble()) {
        return false;
    }

    unsigned char rumble_data[64] = {0};
    if (ctx->device->product_id == USB_PRODUCT_NINTENDO_SWITCH2_GAMECUBE_CONTROLLER) {
        Uint16 rumble_max = SDL_max(ctx->rumble_lo_amp, ctx->rumble_hi_amp);
        rumble_data[0x00] = 0x3;
        rumble_data[1] = 0x50 | (ctx->rumble_seq & 0xf);
        if (rumble_max == 0) {
            rumble_data[2] = 2;
            ctx->rumble_error = 0;
        } else {
            if (ctx->rumble_error < rumble_max) {
                rumble_data[2] = 1;
                ctx->rumble_error += UINT16_MAX - rumble_max;
            } else {
                rumble_data[2] = 0;
                ctx->rumble_error -= rumble_max;
            }
        }
    } else {
        // Rumble can get so strong that it might be dangerous to the controller...
        // This is a game controller, not a massage device, so let's clamp it somewhat
        Uint16 low_amp = (Uint16)((int)ctx->rumble_lo_amp * RUMBLE_MAX / UINT16_MAX);
        Uint16 high_amp = (Uint16)((int)ctx->rumble_hi_amp * RUMBLE_MAX / UINT16_MAX);
        rumble_data[0x01] = 0x50 | (ctx->rumble_seq & 0xf);
        EncodeHDRumble(ctx->rumble_hi_freq, high_amp, ctx->rumble_lo_freq, low_amp, &rumble_data[0x02]);
        switch (ctx->device->product_id) {
        case USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_LEFT:
        case USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_RIGHT:
            if (ctx->device->parent) {
                // FIXME: This shouldn't be necessary, but the rumble thread appears to back up if we don't do this
                interval *= 2;
            }
            rumble_data[0] = 0x1;
            break;
        case USB_PRODUCT_NINTENDO_SWITCH2_PRO:
            rumble_data[0] = 0x2;
            SDL_memcpy(&rumble_data[0x11], &rumble_data[0x01], 6);
            break;
        }
    }
    ctx->rumble_seq++;
    ctx->rumble_updated = false;
    if (!ctx->rumble_lo_amp && !ctx->rumble_hi_amp) {
        ctx->rumble_timestamp = 0;
    } else {
        if (!ctx->rumble_timestamp) {
            ctx->rumble_timestamp = timestamp;
        }
        ctx->rumble_timestamp += interval;
    }

    if (SDL_HIDAPI_SendRumbleAndUnlock(ctx->device, rumble_data, sizeof(rumble_data)) != sizeof(rumble_data)) {
        return SDL_SetError("Couldn't send rumble packet");
    }
    return true;
}

static void HIDAPI_DriverSwitch2_HandleStatePacket(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, SDL_DriverSwitch2_Context *ctx, Uint8 *data, int size)
{
    Uint64 timestamp = SDL_GetTicksNS();

    if (size < 64) {
        // We don't know how to handle this report
        return;
    }

    switch (device->product_id) {
    case USB_PRODUCT_NINTENDO_SWITCH2_GAMECUBE_CONTROLLER:
        HandleGameCubeState(timestamp, joystick, ctx, data, size);
        break;
    case USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_LEFT:
        if (device->parent || ctx->vertical_mode) {
            HandleCombinedControllerStateL(timestamp, joystick, ctx, data, size);
        } else {
            HandleMiniControllerStateL(timestamp, joystick, ctx, data, size);
        }
        break;
    case USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_RIGHT:
        if (device->parent || ctx->vertical_mode) {
            HandleCombinedControllerStateR(timestamp, joystick, ctx, data, size);
        } else {
            HandleMiniControllerStateR(timestamp, joystick, ctx, data, size);
        }
        break;
    case USB_PRODUCT_NINTENDO_SWITCH2_PRO:
        HandleSwitchProState(timestamp, joystick, ctx, data, size);
        break;
    default:
        // FIXME: Need state handling implementation
        break;
    }

    Uint64 sensor_timestamp = (Uint32) (data[0x2b] | (data[0x2c] << 8U) | (data[0x2d] << 16U) | (data[0x2e] << 24U));
    if (sensor_timestamp && !ctx->sensors_ready) {
        ctx->sample_count++;
        if (ctx->sample_count >= 5 && !ctx->first_sensor_timestamp) {
            ctx->first_sensor_timestamp = sensor_timestamp;
            ctx->sample_count = 0;
        } else if (ctx->sample_count == 100) {
            // Calculate timestamp coefficient
            // Timestamp are normally microseconds but sometimes it's something else for no apparent reason
            Uint64 coeff = 1000 * (sensor_timestamp - ctx->first_sensor_timestamp) / (ctx->sample_count * 4);
            if ((coeff + 100000) / 200000 == 5) {
                // Within 10% of 1000
                ctx->sensor_ts_coeff = 10000;
                ctx->gyro_coeff = 34.8f;
                ctx->sensors_ready = true;
            } else if (coeff != 0) {
                ctx->sensor_ts_coeff = 10000000000 / coeff;
                ctx->gyro_coeff = 40.0f;
                ctx->sensors_ready = true;
            } else {
                // Didn't get a valid reading, try again
                ctx->first_sensor_timestamp = 0;
                ctx->sample_count = 0;
            }

            if (ctx->sensors_ready && !ctx->sensors_enabled) {
                Uint8 set_features[] = {
                        0x0c, 0x91, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00
                };
                unsigned char reply[12] = {0};

                SendBulkData(ctx, set_features, sizeof(set_features));
                RecvBulkData(ctx, reply, sizeof(reply));
            }
        }
    }
    if (ctx->sensors_enabled && sensor_timestamp && ctx->sensors_ready) {
        sensor_timestamp = sensor_timestamp * ctx->sensor_ts_coeff / 10;
        float accel_data[3];
        float gyro_data[3];
        const float g = 9.80665f;
        const float accel_scale = g * 8.f / INT16_MAX;

        accel_data[0] = (Sint16)(data[0x31] | (data[0x32] << 8)) * accel_scale;
        accel_data[1] = (Sint16)(data[0x35] | (data[0x36] << 8)) * accel_scale;
        accel_data[2] = (Sint16)(data[0x33] | (data[0x34] << 8)) * -accel_scale;

        gyro_data[0] = (Sint16)(data[0x37] | (data[0x38] << 8)) * ctx->gyro_coeff / INT16_MAX - ctx->gyro_bias_x;
        gyro_data[1] = (Sint16)(data[0x3b] | (data[0x3c] << 8)) * ctx->gyro_coeff / INT16_MAX - ctx->gyro_bias_z;
        gyro_data[2] = (Sint16)(data[0x39] | (data[0x3a] << 8)) * -ctx->gyro_coeff / INT16_MAX + ctx->gyro_bias_y;

        switch (ctx->device->product_id) {
        case USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_LEFT:
            if (ctx->device->parent) {
                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO_L, sensor_timestamp, gyro_data, 3);
                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL_L, sensor_timestamp, accel_data, 3);
            } else {
                float tmp = -accel_data[0];
                accel_data[0] = accel_data[2];
                accel_data[2] = tmp;

                tmp = -gyro_data[0];
                gyro_data[0] = gyro_data[2];
                gyro_data[2] = tmp;

                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, sensor_timestamp, gyro_data, 3);
                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, sensor_timestamp, accel_data, 3);
            }
            break;
        case USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_RIGHT:
            if (ctx->device->parent) {
                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, sensor_timestamp, gyro_data, 3);
                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, sensor_timestamp, accel_data, 3);
                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO_R, sensor_timestamp, gyro_data, 3);
                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL_R, sensor_timestamp, accel_data, 3);
            } else {
                float tmp = accel_data[0];
                accel_data[0] = -accel_data[2];
                accel_data[2] = tmp;

                tmp = gyro_data[0];
                gyro_data[0] = -gyro_data[2];
                gyro_data[2] = tmp;

                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, sensor_timestamp, gyro_data, 3);
                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, sensor_timestamp, accel_data, 3);
            }
            break;
        default:
            SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, sensor_timestamp, gyro_data, 3);
            SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, sensor_timestamp, accel_data, 3);
            break;
        }
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static bool HIDAPI_DriverSwitch2_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSwitch2_Context *ctx = (SDL_DriverSwitch2_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size = 0;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    } else {
        return false;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_SWITCH2_PROTOCOL
        if (device->product_id == USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_LEFT) {
            HIDAPI_DumpPacket("Nintendo Joy-Con(L) packet: size = %d", data, size);
        } else if (device->product_id == USB_PRODUCT_NINTENDO_SWITCH2_JOYCON_RIGHT) {
            HIDAPI_DumpPacket("Nintendo Joy-Con(R) packet: size = %d", data, size);
        } else {
            HIDAPI_DumpPacket("Nintendo Switch2 packet: size = %d", data, size);
        }
#endif
        if (!joystick) {
            continue;
        }

        HIDAPI_DriverSwitch2_HandleStatePacket(device, joystick, ctx, data, size);

        UpdateRumble(ctx);
    }

    if (size < 0) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverSwitch2_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverSwitch2_Context *ctx = (SDL_DriverSwitch2_Context *)device->context;

    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SWITCH_PLAYER_LED,
                           SDL_PlayerLEDHintChanged, ctx);

    ctx->joystick = NULL;
}

static void HIDAPI_DriverSwitch2_FreeDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSwitch2_Context *ctx = (SDL_DriverSwitch2_Context *)device->context;

    if (ctx) {
        if (ctx->interface_claimed) {
            ctx->libusb->release_interface(ctx->device_handle, ctx->interface_number);
            ctx->interface_claimed = false;
        }
        if (ctx->libusb) {
            SDL_QuitLibUSB();
            ctx->libusb = NULL;
        }
    }
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverSwitch2 = {
    SDL_HINT_JOYSTICK_HIDAPI_SWITCH2,
    true,
    HIDAPI_DriverSwitch2_RegisterHints,
    HIDAPI_DriverSwitch2_UnregisterHints,
    HIDAPI_DriverSwitch2_IsEnabled,
    HIDAPI_DriverSwitch2_IsSupportedDevice,
    HIDAPI_DriverSwitch2_InitDevice,
    HIDAPI_DriverSwitch2_GetDevicePlayerIndex,
    HIDAPI_DriverSwitch2_SetDevicePlayerIndex,
    HIDAPI_DriverSwitch2_UpdateDevice,
    HIDAPI_DriverSwitch2_OpenJoystick,
    HIDAPI_DriverSwitch2_RumbleJoystick,
    HIDAPI_DriverSwitch2_RumbleJoystickTriggers,
    HIDAPI_DriverSwitch2_GetJoystickCapabilities,
    HIDAPI_DriverSwitch2_SetJoystickLED,
    HIDAPI_DriverSwitch2_SendJoystickEffect,
    HIDAPI_DriverSwitch2_SetJoystickSensorsEnabled,
    HIDAPI_DriverSwitch2_CloseJoystick,
    HIDAPI_DriverSwitch2_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_SWITCH2

#endif // SDL_JOYSTICK_HIDAPI
