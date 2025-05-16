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

#ifdef SDL_JOYSTICK_HIDAPI

#include "../../SDL_hints_c.h"
#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"

#ifdef SDL_JOYSTICK_HIDAPI_PS3

// Define this if you want to log all packets from the controller
// #define DEBUG_PS3_PROTOCOL

#define LOAD16(A, B) (Sint16)((Uint16)(A) | (((Uint16)(B)) << 8))

typedef enum
{
    k_EPS3ReportIdState = 1,
    k_EPS3ReportIdEffects = 1,
} EPS3ReportId;

typedef enum
{
    k_EPS3SonySixaxisReportIdState = 0,
    k_EPS3SonySixaxisReportIdEffects = 0,
} EPS3SonySixaxisReportId;

// Commands for Sony's sixaxis.sys Windows driver
// All commands must be sent using 49-byte buffer containing output report
// Byte 0 indicates reportId and must always be 0
// Byte 1 indicates a command, supported values are specified below:
typedef enum
{
    // This command allows to set user LEDs.
    // Bytes 5,6.7.8 contain mode for corresponding LED: 0 - LED is off, 1 - LED in on, 2 - LED is flashing.
    // Bytes 9-16 specify 64-bit LED flash period in 100 ns units if some LED is flashing, otherwise not used.
    k_EPS3SixaxisCommandSetLEDs = 1,

    // This command allows to set left and right motors.
    // Byte 5 is right motor duration (0-255) and byte 6, if not zero, activates right motor. Zero value disables right motor.
    // Byte 7 is left motor duration (0-255) and byte 8 is left motor amplitude (0-255).
    k_EPS3SixaxisCommandSetMotors = 2,

    // This command allows to block/unblock setting device LEDs by applications.
    // Byte 5 is used as parameter - any non-zero value blocks LEDs, zero value will unblock LEDs.
    k_EPS3SixaxisCommandBlockLEDs = 3,

    // This command refreshes driver settings. No parameters used.
    // When sixaxis driver loads it reads 'CurrentDriverSetting' binary value from 'HKLM\System\CurrentControlSet\Services\sixaxis\Parameters' registry key.
    // If the key is not present then default values are used. Sending this command forces sixaxis driver to re-read the registry and update driver settings.
    k_EPS3SixaxisCommandRefreshDriverSetting = 9,

    // This command clears current bluetooth pairing. No parameters used.
    k_EPS3SixaxisCommandClearPairing = 10
} EPS3SixaxisDriverCommands;

typedef struct
{
    SDL_HIDAPI_Device *device;
    SDL_Joystick *joystick;
    bool is_shanwan;
    bool has_analog_buttons;
    bool report_sensors;
    bool effects_updated;
    int player_index;
    Uint8 rumble_left;
    Uint8 rumble_right;
    Uint8 last_state[USB_PACKET_LENGTH];
} SDL_DriverPS3_Context;

static bool HIDAPI_DriverPS3_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *effect, int size);

static void HIDAPI_DriverPS3_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_PS3, callback, userdata);
}

static void HIDAPI_DriverPS3_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_PS3, callback, userdata);
}

static bool HIDAPI_DriverPS3_IsEnabled(void)
{
    bool default_value;

#ifdef SDL_PLATFORM_MACOS
    // This works well on macOS
    default_value = true;
#elif defined(SDL_PLATFORM_WIN32)
    /* For official Sony driver (sixaxis.sys) use SDL_HINT_JOYSTICK_HIDAPI_PS3_SIXAXIS_DRIVER.
     *
     * See https://github.com/ViGEm/DsHidMini as an alternative driver
     */
    default_value = false;
#elif defined(SDL_PLATFORM_LINUX)
    /* Linux drivers do a better job of managing the transition between
     * USB and Bluetooth. There are also some quirks in communicating
     * with PS3 controllers that have been implemented in SDL's hidapi
     * for libusb, but are not possible to support using hidraw if the
     * kernel doesn't already know about them.
     */
    default_value = false;
#else
    // Untested, default off
    default_value = false;
#endif

    if (default_value) {
        default_value = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT);
    }
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_PS3, default_value);
}

static bool HIDAPI_DriverPS3_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (vendor_id == USB_VENDOR_SONY && product_id == USB_PRODUCT_SONY_DS3) {
        return true;
    }
    if (vendor_id == USB_VENDOR_SHANWAN && product_id == USB_PRODUCT_SHANWAN_DS3) {
        return true;
    }
    return false;
}

static int ReadFeatureReport(SDL_hid_device *dev, Uint8 report_id, Uint8 *report, size_t length)
{
    SDL_memset(report, 0, length);
    report[0] = report_id;
    return SDL_hid_get_feature_report(dev, report, length);
}

static int SendFeatureReport(SDL_hid_device *dev, Uint8 *report, size_t length)
{
    return SDL_hid_send_feature_report(dev, report, length);
}

static bool HIDAPI_DriverPS3_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx;
    bool is_shanwan = false;

    if (device->vendor_id == USB_VENDOR_SONY &&
        SDL_strncasecmp(device->name, "ShanWan", 7) == 0) {
        is_shanwan = true;
    }
    if (device->vendor_id == USB_VENDOR_SHANWAN ||
        device->vendor_id == USB_VENDOR_SHANWAN_ALT) {
        is_shanwan = true;
    }

    ctx = (SDL_DriverPS3_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    ctx->device = device;
    ctx->is_shanwan = is_shanwan;
    ctx->has_analog_buttons = true;

    device->context = ctx;

    // Set the controller into report mode over Bluetooth
    if (device->is_bluetooth) {
        Uint8 data[] = { 0xf4, 0x42, 0x03, 0x00, 0x00 };

        SendFeatureReport(device->dev, data, sizeof(data));
    }

    // Set the controller into report mode over USB
    if (!device->is_bluetooth) {
        Uint8 data[USB_PACKET_LENGTH];

        int size = ReadFeatureReport(device->dev, 0xf2, data, 17);
        if (size < 0) {
            SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                         "HIDAPI_DriverPS3_InitDevice(): Couldn't read feature report 0xf2");
            return false;
        }
#ifdef DEBUG_PS3_PROTOCOL
        HIDAPI_DumpPacket("PS3 0xF2 packet: size = %d", data, size);
#endif
        size = ReadFeatureReport(device->dev, 0xf5, data, 8);
        if (size < 0) {
            SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                         "HIDAPI_DriverPS3_InitDevice(): Couldn't read feature report 0xf5");
            return false;
        }
#ifdef DEBUG_PS3_PROTOCOL
        HIDAPI_DumpPacket("PS3 0xF5 packet: size = %d", data, size);
#endif
        if (!ctx->is_shanwan) {
            // An output report could cause ShanWan controllers to rumble non-stop
            SDL_hid_write(device->dev, data, 1);
        }
    }

    device->type = SDL_GAMEPAD_TYPE_PS3;
    HIDAPI_SetDeviceName(device, "PS3 Controller");

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverPS3_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static bool HIDAPI_DriverPS3_UpdateEffects(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    Uint8 effects[] = {
        0x01, 0xff, 0x00, 0xff, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0xff, 0x27, 0x10, 0x00, 0x32,
        0xff, 0x27, 0x10, 0x00, 0x32,
        0xff, 0x27, 0x10, 0x00, 0x32,
        0xff, 0x27, 0x10, 0x00, 0x32,
        0x00, 0x00, 0x00, 0x00, 0x00
    };

    effects[2] = ctx->rumble_right ? 1 : 0;
    effects[4] = ctx->rumble_left;

    effects[9] = (0x01 << (1 + (ctx->player_index % 4)));

    return HIDAPI_DriverPS3_SendJoystickEffect(device, ctx->joystick, effects, sizeof(effects));
}

static void HIDAPI_DriverPS3_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    if (!ctx) {
        return;
    }

    ctx->player_index = player_index;

    // This will set the new LED state based on the new player index
    HIDAPI_DriverPS3_UpdateEffects(device);
}

static bool HIDAPI_DriverPS3_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->joystick = joystick;
    ctx->effects_updated = false;
    ctx->rumble_left = 0;
    ctx->rumble_right = 0;
    SDL_zeroa(ctx->last_state);

    // Initialize player index (needed for setting LEDs)
    ctx->player_index = SDL_GetJoystickPlayerIndex(joystick);

    // Initialize the joystick capabilities
    joystick->nbuttons = 11;
    joystick->naxes = 6;
    if (ctx->has_analog_buttons) {
        joystick->naxes += 10;
    }
    joystick->nhats = 1;

    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, 100.0f);

    return true;
}

static bool HIDAPI_DriverPS3_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->rumble_left = (low_frequency_rumble >> 8);
    ctx->rumble_right = (high_frequency_rumble >> 8);

    return HIDAPI_DriverPS3_UpdateEffects(device);
}

static bool HIDAPI_DriverPS3_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverPS3_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    return SDL_JOYSTICK_CAP_RUMBLE;
}

static bool HIDAPI_DriverPS3_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverPS3_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *effect, int size)
{
    Uint8 data[49];
    int report_size, offset;

    SDL_zeroa(data);

    data[0] = k_EPS3ReportIdEffects;
    report_size = sizeof(data);
    offset = 1;
    SDL_memcpy(&data[offset], effect, SDL_min((sizeof(data) - offset), (size_t)size));

    if (SDL_HIDAPI_SendRumble(device, data, report_size) != report_size) {
        return SDL_SetError("Couldn't send rumble packet");
    }
    return true;
}

static bool HIDAPI_DriverPS3_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->report_sensors = enabled;

    return true;
}

static float HIDAPI_DriverPS3_ScaleAccel(Sint16 value)
{
    // Accelerometer values are in big endian order
    value = SDL_Swap16BE(value);
    return ((float)(value - 511) / 113.0f) * SDL_STANDARD_GRAVITY;
}

static void HIDAPI_DriverPS3_HandleMiniStatePacket(SDL_Joystick *joystick, SDL_DriverPS3_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

    if (ctx->last_state[4] != data[4]) {
        Uint8 hat;

        switch (data[4] & 0x0f) {
        case 0:
            hat = SDL_HAT_UP;
            break;
        case 1:
            hat = SDL_HAT_RIGHTUP;
            break;
        case 2:
            hat = SDL_HAT_RIGHT;
            break;
        case 3:
            hat = SDL_HAT_RIGHTDOWN;
            break;
        case 4:
            hat = SDL_HAT_DOWN;
            break;
        case 5:
            hat = SDL_HAT_LEFTDOWN;
            break;
        case 6:
            hat = SDL_HAT_LEFT;
            break;
        case 7:
            hat = SDL_HAT_LEFTUP;
            break;
        default:
            hat = SDL_HAT_CENTERED;
            break;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[4] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[4] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[4] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[4] & 0x80) != 0));
    }

    if (ctx->last_state[5] != data[5]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[5] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[5] & 0x02) != 0));
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, (data[5] & 0x04) ? SDL_JOYSTICK_AXIS_MAX : SDL_JOYSTICK_AXIS_MIN);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, (data[5] & 0x08) ? SDL_JOYSTICK_AXIS_MAX : SDL_JOYSTICK_AXIS_MIN);
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[5] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[5] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[5] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[5] & 0x80) != 0));
    }

    axis = ((int)data[2] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
    axis = ((int)data[3] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
    axis = ((int)data[0] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
    axis = ((int)data[1] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static void HIDAPI_DriverPS3_HandleStatePacket(SDL_Joystick *joystick, SDL_DriverPS3_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

    if (ctx->last_state[2] != data[2]) {
        Uint8 hat = 0;

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[2] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[2] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[2] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[2] & 0x08) != 0));

        if (data[2] & 0x10) {
            hat |= SDL_HAT_UP;
        }
        if (data[2] & 0x20) {
            hat |= SDL_HAT_RIGHT;
        }
        if (data[2] & 0x40) {
            hat |= SDL_HAT_DOWN;
        }
        if (data[2] & 0x80) {
            hat |= SDL_HAT_LEFT;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);
    }

    if (ctx->last_state[3] != data[3]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[3] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[3] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[3] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[3] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[3] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[3] & 0x80) != 0));
    }

    if (ctx->last_state[4] != data[4]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[4] & 0x01) != 0));
    }

    axis = ((int)data[18] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
    axis = ((int)data[19] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);
    axis = ((int)data[6] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
    axis = ((int)data[7] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
    axis = ((int)data[8] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
    axis = ((int)data[9] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);

    // Buttons are mapped as axes in the order they appear in the button enumeration
    if (ctx->has_analog_buttons) {
        static int button_axis_offsets[] = {
            24, // SDL_GAMEPAD_BUTTON_SOUTH
            23, // SDL_GAMEPAD_BUTTON_EAST
            25, // SDL_GAMEPAD_BUTTON_WEST
            22, // SDL_GAMEPAD_BUTTON_NORTH
            0,  // SDL_GAMEPAD_BUTTON_BACK
            0,  // SDL_GAMEPAD_BUTTON_GUIDE
            0,  // SDL_GAMEPAD_BUTTON_START
            0,  // SDL_GAMEPAD_BUTTON_LEFT_STICK
            0,  // SDL_GAMEPAD_BUTTON_RIGHT_STICK
            20, // SDL_GAMEPAD_BUTTON_LEFT_SHOULDER
            21, // SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER
            14, // SDL_GAMEPAD_BUTTON_DPAD_UP
            16, // SDL_GAMEPAD_BUTTON_DPAD_DOWN
            17, // SDL_GAMEPAD_BUTTON_DPAD_LEFT
            15, // SDL_GAMEPAD_BUTTON_DPAD_RIGHT
        };
        Uint8 i, axis_index = 6;

        for (i = 0; i < SDL_arraysize(button_axis_offsets); ++i) {
            int offset = button_axis_offsets[i];
            if (!offset) {
                // This button doesn't report as an axis
                continue;
            }

            axis = ((int)data[offset] * 257) - 32768;
            SDL_SendJoystickAxis(timestamp, joystick, axis_index, axis);
            ++axis_index;
        }
    }

    if (ctx->report_sensors) {
        float sensor_data[3];

        sensor_data[0] = HIDAPI_DriverPS3_ScaleAccel(LOAD16(data[41], data[42]));
        sensor_data[1] = -HIDAPI_DriverPS3_ScaleAccel(LOAD16(data[45], data[46]));
        sensor_data[2] = -HIDAPI_DriverPS3_ScaleAccel(LOAD16(data[43], data[44]));
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, timestamp, sensor_data, SDL_arraysize(sensor_data));
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static bool HIDAPI_DriverPS3_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    } else {
        return false;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_PS3_PROTOCOL
        HIDAPI_DumpPacket("PS3 packet: size = %d", data, size);
#endif
        if (!joystick) {
            continue;
        }

        if (size == 7) {
            // Seen on a ShanWan PS2 -> PS3 USB converter
            HIDAPI_DriverPS3_HandleMiniStatePacket(joystick, ctx, data, size);

            // Wait for the first report to set the LED state after the controller stops blinking
            if (!ctx->effects_updated) {
                HIDAPI_DriverPS3_UpdateEffects(device);
                ctx->effects_updated = true;
            }
            continue;
        }

        switch (data[0]) {
        case k_EPS3ReportIdState:
            if (data[1] == 0xFF) {
                // Invalid data packet, ignore
                break;
            }
            HIDAPI_DriverPS3_HandleStatePacket(joystick, ctx, data, size);

            // Wait for the first report to set the LED state after the controller stops blinking
            if (!ctx->effects_updated) {
                HIDAPI_DriverPS3_UpdateEffects(device);
                ctx->effects_updated = true;
            }
            break;
        default:
#ifdef DEBUG_JOYSTICK
            SDL_Log("Unknown PS3 packet: 0x%.2x", data[0]);
#endif
            break;
        }
    }

    if (size < 0) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverPS3_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->joystick = NULL;
}

static void HIDAPI_DriverPS3_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS3 = {
    SDL_HINT_JOYSTICK_HIDAPI_PS3,
    true,
    HIDAPI_DriverPS3_RegisterHints,
    HIDAPI_DriverPS3_UnregisterHints,
    HIDAPI_DriverPS3_IsEnabled,
    HIDAPI_DriverPS3_IsSupportedDevice,
    HIDAPI_DriverPS3_InitDevice,
    HIDAPI_DriverPS3_GetDevicePlayerIndex,
    HIDAPI_DriverPS3_SetDevicePlayerIndex,
    HIDAPI_DriverPS3_UpdateDevice,
    HIDAPI_DriverPS3_OpenJoystick,
    HIDAPI_DriverPS3_RumbleJoystick,
    HIDAPI_DriverPS3_RumbleJoystickTriggers,
    HIDAPI_DriverPS3_GetJoystickCapabilities,
    HIDAPI_DriverPS3_SetJoystickLED,
    HIDAPI_DriverPS3_SendJoystickEffect,
    HIDAPI_DriverPS3_SetJoystickSensorsEnabled,
    HIDAPI_DriverPS3_CloseJoystick,
    HIDAPI_DriverPS3_FreeDevice,
};

static bool HIDAPI_DriverPS3ThirdParty_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_PS3,
                              SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI,
                                                 SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverPS3ThirdParty_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    Uint8 data[USB_PACKET_LENGTH];
    int size;

    if (vendor_id == USB_VENDOR_LOGITECH &&
        product_id == USB_PRODUCT_LOGITECH_CHILLSTREAM) {
        return true;
    }

    if ((type == SDL_GAMEPAD_TYPE_PS3 && vendor_id != USB_VENDOR_SONY) ||
        HIDAPI_SupportsPlaystationDetection(vendor_id, product_id)) {
        if (device && device->dev) {
            size = ReadFeatureReport(device->dev, 0x03, data, sizeof(data));
            if (size == 8 && data[2] == 0x26) {
                // Supported third party controller
                return true;
            } else {
                return false;
            }
        } else {
            // Might be supported by this driver, enumerate and find out
            return true;
        }
    }
    return false;
}

static bool HIDAPI_DriverPS3ThirdParty_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx;

    ctx = (SDL_DriverPS3_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    ctx->device = device;
    if (device->vendor_id == USB_VENDOR_SWITCH && device->product_id == USB_PRODUCT_SWITCH_RETROBIT_CONTROLLER) {
        ctx->has_analog_buttons = false;
    } else {
        ctx->has_analog_buttons = true;
    }

    device->context = ctx;

    device->type = SDL_GAMEPAD_TYPE_PS3;

    if (device->vendor_id == USB_VENDOR_LOGITECH &&
        device->product_id == USB_PRODUCT_LOGITECH_CHILLSTREAM) {
        HIDAPI_SetDeviceName(device, "Logitech ChillStream");
    }

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverPS3ThirdParty_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverPS3ThirdParty_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static bool HIDAPI_DriverPS3ThirdParty_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->joystick = joystick;
    SDL_zeroa(ctx->last_state);

    // Initialize the joystick capabilities
    joystick->nbuttons = 11;
    joystick->naxes = 6;
    if (ctx->has_analog_buttons) {
        joystick->naxes += 10;
    }
    joystick->nhats = 1;

    if (device->vendor_id == USB_VENDOR_SWITCH && device->product_id == USB_PRODUCT_SWITCH_RETROBIT_CONTROLLER) {
        // This is a wireless controller using a USB dongle
        joystick->connection_state = SDL_JOYSTICK_CONNECTION_WIRELESS;
    }

    return true;
}

static bool HIDAPI_DriverPS3ThirdParty_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverPS3ThirdParty_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverPS3ThirdParty_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    return 0;
}

static bool HIDAPI_DriverPS3ThirdParty_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverPS3ThirdParty_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *effect, int size)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverPS3ThirdParty_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static void HIDAPI_DriverPS3ThirdParty_HandleStatePacket18(SDL_Joystick *joystick, SDL_DriverPS3_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

    if (ctx->last_state[0] != data[0]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[0] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[0] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[0] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[0] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[0] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[0] & 0x20) != 0));
    }

    if (ctx->last_state[1] != data[1]) {
        Uint8 hat;

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[1] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[1] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[1] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[1] & 0x08) != 0));

        switch (data[1] >> 4) {
        case 0:
            hat = SDL_HAT_UP;
            break;
        case 1:
            hat = SDL_HAT_RIGHTUP;
            break;
        case 2:
            hat = SDL_HAT_RIGHT;
            break;
        case 3:
            hat = SDL_HAT_RIGHTDOWN;
            break;
        case 4:
            hat = SDL_HAT_DOWN;
            break;
        case 5:
            hat = SDL_HAT_LEFTDOWN;
            break;
        case 6:
            hat = SDL_HAT_LEFT;
            break;
        case 7:
            hat = SDL_HAT_LEFTUP;
            break;
        default:
            hat = SDL_HAT_CENTERED;
            break;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);
    }

    axis = ((int)data[16] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
    axis = ((int)data[17] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);
    axis = ((int)data[2] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
    axis = ((int)data[3] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
    axis = ((int)data[4] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
    axis = ((int)data[5] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);

    // Buttons are mapped as axes in the order they appear in the button enumeration
    if (ctx->has_analog_buttons) {
        static int button_axis_offsets[] = {
            12, // SDL_GAMEPAD_BUTTON_SOUTH
            11, // SDL_GAMEPAD_BUTTON_EAST
            13, // SDL_GAMEPAD_BUTTON_WEST
            10, // SDL_GAMEPAD_BUTTON_NORTH
            0,  // SDL_GAMEPAD_BUTTON_BACK
            0,  // SDL_GAMEPAD_BUTTON_GUIDE
            0,  // SDL_GAMEPAD_BUTTON_START
            0,  // SDL_GAMEPAD_BUTTON_LEFT_STICK
            0,  // SDL_GAMEPAD_BUTTON_RIGHT_STICK
            14, // SDL_GAMEPAD_BUTTON_LEFT_SHOULDER
            15, // SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER
            8,  // SDL_GAMEPAD_BUTTON_DPAD_UP
            9,  // SDL_GAMEPAD_BUTTON_DPAD_DOWN
            7,  // SDL_GAMEPAD_BUTTON_DPAD_LEFT
            6,  // SDL_GAMEPAD_BUTTON_DPAD_RIGHT
        };
        Uint8 i, axis_index = 6;

        for (i = 0; i < SDL_arraysize(button_axis_offsets); ++i) {
            int offset = button_axis_offsets[i];
            if (!offset) {
                // This button doesn't report as an axis
                continue;
            }

            axis = ((int)data[offset] * 257) - 32768;
            SDL_SendJoystickAxis(timestamp, joystick, axis_index, axis);
            ++axis_index;
        }
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static void HIDAPI_DriverPS3ThirdParty_HandleStatePacket19(SDL_Joystick *joystick, SDL_DriverPS3_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

    if (ctx->last_state[0] != data[0]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[0] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[0] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[0] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[0] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[0] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[0] & 0x20) != 0));
    }

    if (ctx->last_state[1] != data[1]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[1] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[1] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[1] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[1] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[1] & 0x10) != 0));
    }

    if (ctx->device->vendor_id == USB_VENDOR_SAITEK && ctx->device->product_id == USB_PRODUCT_SAITEK_CYBORG_V3) {
        // Cyborg V.3 Rumble Pad doesn't set the dpad bits as expected, so use the axes instead
        Uint8 hat = 0;

        if (data[7]) {
            hat |= SDL_HAT_RIGHT;
        }
        if (data[8]) {
            hat |= SDL_HAT_LEFT;
        }
        if (data[9]) {
            hat |= SDL_HAT_UP;
        }
        if (data[10]) {
            hat |= SDL_HAT_DOWN;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);
    } else {
        if (ctx->last_state[2] != data[2]) {
            Uint8 hat;

            switch (data[2] & 0x0f) {
            case 0:
                hat = SDL_HAT_UP;
                break;
            case 1:
                hat = SDL_HAT_RIGHTUP;
                break;
            case 2:
                hat = SDL_HAT_RIGHT;
                break;
            case 3:
                hat = SDL_HAT_RIGHTDOWN;
                break;
            case 4:
                hat = SDL_HAT_DOWN;
                break;
            case 5:
                hat = SDL_HAT_LEFTDOWN;
                break;
            case 6:
                hat = SDL_HAT_LEFT;
                break;
            case 7:
                hat = SDL_HAT_LEFTUP;
                break;
            default:
                hat = SDL_HAT_CENTERED;
                break;
            }
            SDL_SendJoystickHat(timestamp, joystick, 0, hat);
        }
    }

    if (data[0] & 0x40) {
        axis = 32767;
    } else {
        axis = ((int)data[17] * 257) - 32768;
    }
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
    if (data[0] & 0x80) {
        axis = 32767;
    } else {
        axis = ((int)data[18] * 257) - 32768;
    }
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);
    axis = ((int)data[3] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
    axis = ((int)data[4] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
    axis = ((int)data[5] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
    axis = ((int)data[6] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);

    // Buttons are mapped as axes in the order they appear in the button enumeration
    if (ctx->has_analog_buttons) {
        static int button_axis_offsets[] = {
            13, // SDL_GAMEPAD_BUTTON_SOUTH
            12, // SDL_GAMEPAD_BUTTON_EAST
            14, // SDL_GAMEPAD_BUTTON_WEST
            11, // SDL_GAMEPAD_BUTTON_NORTH
            0,  // SDL_GAMEPAD_BUTTON_BACK
            0,  // SDL_GAMEPAD_BUTTON_GUIDE
            0,  // SDL_GAMEPAD_BUTTON_START
            0,  // SDL_GAMEPAD_BUTTON_LEFT_STICK
            0,  // SDL_GAMEPAD_BUTTON_RIGHT_STICK
            15, // SDL_GAMEPAD_BUTTON_LEFT_SHOULDER
            16, // SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER
            9,  // SDL_GAMEPAD_BUTTON_DPAD_UP
            10, // SDL_GAMEPAD_BUTTON_DPAD_DOWN
            8,  // SDL_GAMEPAD_BUTTON_DPAD_LEFT
            7,  // SDL_GAMEPAD_BUTTON_DPAD_RIGHT
        };
        Uint8 i, axis_index = 6;

        for (i = 0; i < SDL_arraysize(button_axis_offsets); ++i) {
            int offset = button_axis_offsets[i];
            if (!offset) {
                // This button doesn't report as an axis
                continue;
            }

            axis = ((int)data[offset] * 257) - 32768;
            SDL_SendJoystickAxis(timestamp, joystick, axis_index, axis);
            ++axis_index;
        }
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static bool HIDAPI_DriverPS3ThirdParty_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    } else {
        return false;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_PS3_PROTOCOL
        HIDAPI_DumpPacket("PS3 packet: size = %d", data, size);
#endif
        if (!joystick) {
            continue;
        }

        if (size >= 19) {
            HIDAPI_DriverPS3ThirdParty_HandleStatePacket19(joystick, ctx, data, size);
        } else if (size == 18) {
            // This packet format was seen with the Logitech ChillStream
            HIDAPI_DriverPS3ThirdParty_HandleStatePacket18(joystick, ctx, data, size);
        } else {
#ifdef DEBUG_JOYSTICK
            SDL_Log("Unknown PS3 packet, size %d", size);
#endif
        }
    }

    if (size < 0) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverPS3ThirdParty_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->joystick = NULL;
}

static void HIDAPI_DriverPS3ThirdParty_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS3ThirdParty = {
    SDL_HINT_JOYSTICK_HIDAPI_PS3,
    true,
    HIDAPI_DriverPS3_RegisterHints,
    HIDAPI_DriverPS3_UnregisterHints,
    HIDAPI_DriverPS3ThirdParty_IsEnabled,
    HIDAPI_DriverPS3ThirdParty_IsSupportedDevice,
    HIDAPI_DriverPS3ThirdParty_InitDevice,
    HIDAPI_DriverPS3ThirdParty_GetDevicePlayerIndex,
    HIDAPI_DriverPS3ThirdParty_SetDevicePlayerIndex,
    HIDAPI_DriverPS3ThirdParty_UpdateDevice,
    HIDAPI_DriverPS3ThirdParty_OpenJoystick,
    HIDAPI_DriverPS3ThirdParty_RumbleJoystick,
    HIDAPI_DriverPS3ThirdParty_RumbleJoystickTriggers,
    HIDAPI_DriverPS3ThirdParty_GetJoystickCapabilities,
    HIDAPI_DriverPS3ThirdParty_SetJoystickLED,
    HIDAPI_DriverPS3ThirdParty_SendJoystickEffect,
    HIDAPI_DriverPS3ThirdParty_SetJoystickSensorsEnabled,
    HIDAPI_DriverPS3ThirdParty_CloseJoystick,
    HIDAPI_DriverPS3ThirdParty_FreeDevice,
};

static bool HIDAPI_DriverPS3_UpdateRumbleSonySixaxis(SDL_HIDAPI_Device *device);
static bool HIDAPI_DriverPS3_UpdateLEDsSonySixaxis(SDL_HIDAPI_Device *device);

static void HIDAPI_DriverPS3SonySixaxis_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_PS3_SIXAXIS_DRIVER, callback, userdata);
}

static void HIDAPI_DriverPS3SonySixaxis_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_PS3_SIXAXIS_DRIVER, callback, userdata);
}

static bool HIDAPI_DriverPS3SonySixaxis_IsEnabled(void)
{
#ifdef SDL_PLATFORM_WIN32
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_PS3_SIXAXIS_DRIVER, false);
#else
    return false;
#endif
}

static bool HIDAPI_DriverPS3SonySixaxis_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (vendor_id == USB_VENDOR_SONY && product_id == USB_PRODUCT_SONY_DS3) {
        return true;
    }
    return false;
}

static bool HIDAPI_DriverPS3SonySixaxis_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx;

    ctx = (SDL_DriverPS3_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    ctx->device = device;
    ctx->has_analog_buttons = true;

    device->context = ctx;

    Uint8 data[USB_PACKET_LENGTH];

    int size = ReadFeatureReport(device->dev, 0xf2, data, sizeof(data));
    if (size < 0) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                     "HIDAPI_DriverPS3SonySixaxis_InitDevice(): Couldn't read feature report 0xf2. Trying again with 0x0.");
        SDL_zeroa(data);
        size = ReadFeatureReport(device->dev, 0x00, data, sizeof(data));
        if (size < 0) {
            SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                         "HIDAPI_DriverPS3SonySixaxis_InitDevice(): Couldn't read feature report 0x00.");
            return false;
        }
#ifdef DEBUG_PS3_PROTOCOL
        HIDAPI_DumpPacket("PS3 0x0 packet: size = %d", data, size);
#endif
    }
#ifdef DEBUG_PS3_PROTOCOL
    HIDAPI_DumpPacket("PS3 0xF2 packet: size = %d", data, size);
#endif

    device->type = SDL_GAMEPAD_TYPE_PS3;
    HIDAPI_SetDeviceName(device, "PS3 Controller");

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverPS3SonySixaxis_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverPS3SonySixaxis_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    if (!ctx) {
        return;
    }

    ctx->player_index = player_index;

    // This will set the new LED state based on the new player index
    HIDAPI_DriverPS3_UpdateLEDsSonySixaxis(device);
}

static bool HIDAPI_DriverPS3SonySixaxis_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->joystick = joystick;
    ctx->effects_updated = false;
    ctx->rumble_left = 0;
    ctx->rumble_right = 0;
    SDL_zeroa(ctx->last_state);

    // Initialize player index (needed for setting LEDs)
    ctx->player_index = SDL_GetJoystickPlayerIndex(joystick);

    // Initialize the joystick capabilities
    joystick->nbuttons = 11;
    joystick->naxes = 6;
    if (ctx->has_analog_buttons) {
        joystick->naxes += 10;
    }
    joystick->nhats = 1;

    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, 100.0f);

    return true;
}

static bool HIDAPI_DriverPS3SonySixaxis_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->rumble_left = (low_frequency_rumble >> 8);
    ctx->rumble_right = (high_frequency_rumble >> 8);

    return HIDAPI_DriverPS3_UpdateRumbleSonySixaxis(device);
}

static bool HIDAPI_DriverPS3SonySixaxis_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverPS3SonySixaxis_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    return 0;
}

static bool HIDAPI_DriverPS3SonySixaxis_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverPS3SonySixaxis_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *effect, int size)
{
    Uint8 data[49];
    int report_size;

    SDL_zeroa(data);

    data[0] = k_EPS3SonySixaxisReportIdEffects;
    report_size = sizeof(data);

    // No offset with Sony sixaxis.sys driver
    SDL_memcpy(&data, effect, SDL_min(sizeof(data), (size_t)size));

    if (SDL_HIDAPI_SendRumble(device, data, report_size) != report_size) {
        return SDL_SetError("Couldn't send rumble packet");
    }
    return true;
}

static bool HIDAPI_DriverPS3SonySixaxis_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->report_sensors = enabled;

    return true;
}

static void HIDAPI_DriverPS3SonySixaxis_HandleStatePacket(SDL_Joystick *joystick, SDL_DriverPS3_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

    if (ctx->last_state[2] != data[2]) {
        Uint8 hat = 0;

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[2] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[2] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[2] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[2] & 0x08) != 0));

        if (data[2] & 0x10) {
            hat |= SDL_HAT_UP;
        }
        if (data[2] & 0x20) {
            hat |= SDL_HAT_RIGHT;
        }
        if (data[2] & 0x40) {
            hat |= SDL_HAT_DOWN;
        }
        if (data[2] & 0x80) {
            hat |= SDL_HAT_LEFT;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);
    }

    if (ctx->last_state[3] != data[3]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[3] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[3] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[3] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[3] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[3] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[3] & 0x80) != 0));
    }

    if (ctx->last_state[4] != data[4]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[4] & 0x01) != 0));
    }

    axis = ((int)data[18] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
    axis = ((int)data[19] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);
    axis = ((int)data[6] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
    axis = ((int)data[7] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
    axis = ((int)data[8] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
    axis = ((int)data[9] * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);

    // Buttons are mapped as axes in the order they appear in the button enumeration
    if (ctx->has_analog_buttons) {
        static int button_axis_offsets[] = {
            24, // SDL_GAMEPAD_BUTTON_SOUTH
            23, // SDL_GAMEPAD_BUTTON_EAST
            25, // SDL_GAMEPAD_BUTTON_WEST
            22, // SDL_GAMEPAD_BUTTON_NORTH
            0,  // SDL_GAMEPAD_BUTTON_BACK
            0,  // SDL_GAMEPAD_BUTTON_GUIDE
            0,  // SDL_GAMEPAD_BUTTON_START
            0,  // SDL_GAMEPAD_BUTTON_LEFT_STICK
            0,  // SDL_GAMEPAD_BUTTON_RIGHT_STICK
            20, // SDL_GAMEPAD_BUTTON_LEFT_SHOULDER
            21, // SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER
            14, // SDL_GAMEPAD_BUTTON_DPAD_UP
            16, // SDL_GAMEPAD_BUTTON_DPAD_DOWN
            17, // SDL_GAMEPAD_BUTTON_DPAD_LEFT
            15, // SDL_GAMEPAD_BUTTON_DPAD_RIGHT
        };
        Uint8 i, axis_index = 6;

        for (i = 0; i < SDL_arraysize(button_axis_offsets); ++i) {
            int offset = button_axis_offsets[i];
            if (!offset) {
                // This button doesn't report as an axis
                continue;
            }

            axis = ((int)data[offset] * 257) - 32768;
            SDL_SendJoystickAxis(timestamp, joystick, axis_index, axis);
            ++axis_index;
        }
    }

    if (ctx->report_sensors) {
        float sensor_data[3];

        sensor_data[0] = HIDAPI_DriverPS3_ScaleAccel(LOAD16(data[41], data[42]));
        sensor_data[1] = -HIDAPI_DriverPS3_ScaleAccel(LOAD16(data[45], data[46]));
        sensor_data[2] = -HIDAPI_DriverPS3_ScaleAccel(LOAD16(data[43], data[44]));
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, timestamp, sensor_data, SDL_arraysize(sensor_data));
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static bool HIDAPI_DriverPS3SonySixaxis_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    } else {
        return false;
    }

    if (!joystick) {
        return false;
    }

    // With sixaxis.sys driver we need to use hid_get_feature_report instead of hid_read
    size = ReadFeatureReport(device->dev, 0x0, data, sizeof(data));
    if (size < 0) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                     "HIDAPI_DriverPS3SonySixaxis_UpdateDevice(): Couldn't read feature report 0x00");
        return false;
    }

    switch (data[0]) {
    case k_EPS3SonySixaxisReportIdState:
        HIDAPI_DriverPS3SonySixaxis_HandleStatePacket(joystick, ctx, &data[1], size - 1); // report data starts in data[1]

        // Wait for the first report to set the LED state after the controller stops blinking
        if (!ctx->effects_updated) {
            HIDAPI_DriverPS3_UpdateLEDsSonySixaxis(device);
            ctx->effects_updated = true;
        }

        break;
    default:
#ifdef DEBUG_JOYSTICK
        SDL_Log("Unknown PS3 packet: 0x%.2x", data[0]);
#endif
        break;
    }

    if (size < 0) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverPS3SonySixaxis_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    ctx->joystick = NULL;
}

static void HIDAPI_DriverPS3SonySixaxis_FreeDevice(SDL_HIDAPI_Device *device)
{
}

static bool HIDAPI_DriverPS3_UpdateRumbleSonySixaxis(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    Uint8 effects[] = {
        0x0,                           // Report Id
        k_EPS3SixaxisCommandSetMotors, // 2 = Set Motors
        0x00, 0x00, 0x00,              // padding
        0xff,                          // Small Motor duration - 0xff is forever
        0x00,                          // Small Motor off/on (0 or 1)
        0xff,                          // Large Motor duration - 0xff is forever
        0x00                           // Large Motor force (0 to 255)
    };

    effects[6] = ctx->rumble_right ? 1 : 0; // Small motor
    effects[8] = ctx->rumble_left;          // Large motor

    return HIDAPI_DriverPS3SonySixaxis_SendJoystickEffect(device, ctx->joystick, effects, sizeof(effects));
}

static bool HIDAPI_DriverPS3_UpdateLEDsSonySixaxis(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS3_Context *ctx = (SDL_DriverPS3_Context *)device->context;

    Uint8 effects[] = {
        0x0,                         // Report Id
        k_EPS3SixaxisCommandSetLEDs, // 1 = Set LEDs
        0x00, 0x00, 0x00,            // padding
        0x00, 0x00, 0x00, 0x00       // LED #4, LED #3, LED #2, LED #1 (0 = Off, 1 = On, 2 = Flashing)
    };

    // Turn on LED light on DS3 Controller for relevant player (player_index 0 lights up LED #1, player_index 1 lights up LED #2, etc)
    if (ctx->player_index < 4) {
        effects[8 - ctx->player_index] = 1;
    }

    return HIDAPI_DriverPS3SonySixaxis_SendJoystickEffect(device, ctx->joystick, effects, sizeof(effects));
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS3SonySixaxis = {
    SDL_HINT_JOYSTICK_HIDAPI_PS3_SIXAXIS_DRIVER,
    true,
    HIDAPI_DriverPS3SonySixaxis_RegisterHints,
    HIDAPI_DriverPS3SonySixaxis_UnregisterHints,
    HIDAPI_DriverPS3SonySixaxis_IsEnabled,
    HIDAPI_DriverPS3SonySixaxis_IsSupportedDevice,
    HIDAPI_DriverPS3SonySixaxis_InitDevice,
    HIDAPI_DriverPS3SonySixaxis_GetDevicePlayerIndex,
    HIDAPI_DriverPS3SonySixaxis_SetDevicePlayerIndex,
    HIDAPI_DriverPS3SonySixaxis_UpdateDevice,
    HIDAPI_DriverPS3SonySixaxis_OpenJoystick,
    HIDAPI_DriverPS3SonySixaxis_RumbleJoystick,
    HIDAPI_DriverPS3SonySixaxis_RumbleJoystickTriggers,
    HIDAPI_DriverPS3SonySixaxis_GetJoystickCapabilities,
    HIDAPI_DriverPS3SonySixaxis_SetJoystickLED,
    HIDAPI_DriverPS3SonySixaxis_SendJoystickEffect,
    HIDAPI_DriverPS3SonySixaxis_SetJoystickSensorsEnabled,
    HIDAPI_DriverPS3SonySixaxis_CloseJoystick,
    HIDAPI_DriverPS3SonySixaxis_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_PS3

#endif // SDL_JOYSTICK_HIDAPI
