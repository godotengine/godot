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
#include "SDL_internal.h"

#ifdef SDL_JOYSTICK_HIDAPI

#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"

#ifdef SDL_JOYSTICK_HIDAPI_8BITDO

// Define this if you want to log all packets from the controller
#if 0
#define DEBUG_8BITDO_PROTOCOL
#endif

enum
{
    SDL_GAMEPAD_BUTTON_8BITDO_L4 = 11,
    SDL_GAMEPAD_BUTTON_8BITDO_R4,
    SDL_GAMEPAD_BUTTON_8BITDO_PL,
    SDL_GAMEPAD_BUTTON_8BITDO_PR,
    SDL_GAMEPAD_NUM_8BITDO_BUTTONS,
};

#define SDL_8BITDO_FEATURE_REPORTID_ENABLE_SDL_REPORTID         0x06
#define SDL_8BITDO_REPORTID_SDL_REPORTID                        0x04
#define SDL_8BITDO_REPORTID_NOT_SUPPORTED_SDL_REPORTID          0x03
#define SDL_8BITDO_BT_REPORTID_SDL_REPORTID                     0x01

#define SDL_8BITDO_SENSOR_TIMESTAMP_ENABLE                      0xAA
#define ABITDO_ACCEL_SCALE 4096.f
#define ABITDO_GYRO_MAX_DEGREES_PER_SECOND 2000.f


#define LOAD32(A, B, C, D) ((((Uint32)(A)) << 0) |  \
                            (((Uint32)(B)) << 8) |  \
                            (((Uint32)(C)) << 16) | \
                            (((Uint32)(D)) << 24))


typedef struct
{
    bool sensors_supported;
    bool sensors_enabled;
    bool touchpad_01_supported;
    bool touchpad_02_supported;
    bool rumble_supported;
    bool rumble_type;
    bool rgb_supported;
    bool player_led_supported;
    bool powerstate_supported;
    bool sensor_timestamp_supported;
    Uint8 serial[6];
    Uint16 version;
    Uint16 version_beta;
    float accelScale;
    float gyroScale;
    Uint8 last_state[USB_PACKET_LENGTH];
    Uint64 sensor_timestamp; // Nanoseconds.  Simulate onboard clock. Different models have different rates vs different connection styles.
    Uint64 sensor_timestamp_interval;
    Uint32 last_tick;
} SDL_Driver8BitDo_Context;

#pragma pack(push,1)
typedef struct
{
    bool sensors_supported;
    bool touchpad_01_supported;
    bool touchpad_02_supported;
    bool rumble_supported;
    bool rumble_type;
    bool rgb_supported;
    Uint8 device_type;
    Uint8 serial[6];
    Uint16 version;
    Uint16 version_beta;
    Uint16 pid;
} ABITDO_DEVICE_INFO;

typedef struct
{
    // Accelerometer values
    short sAccelX;
    short sAccelY;
    short sAccelZ;

    // Gyroscope values
    short sGyroX;
    short sGyroY;
    short sGyroZ;
} ABITDO_SENSORS;

#pragma pack(pop)


static void HIDAPI_Driver8BitDo_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_8BITDO, callback, userdata);
}

static void HIDAPI_Driver8BitDo_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_8BITDO, callback, userdata);
}

static bool HIDAPI_Driver8BitDo_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_8BITDO, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static int ReadFeatureReport(SDL_hid_device *dev, Uint8 report_id, Uint8 *report, size_t length)
{
    SDL_memset(report, 0, length);
    report[0] = report_id;
    return SDL_hid_get_feature_report(dev, report, length);
}

static bool HIDAPI_Driver8BitDo_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (vendor_id == USB_VENDOR_8BITDO) {
        switch (product_id) {
        case USB_PRODUCT_8BITDO_SF30_PRO:
        case USB_PRODUCT_8BITDO_SF30_PRO_BT:
        case USB_PRODUCT_8BITDO_SN30_PRO:
        case USB_PRODUCT_8BITDO_SN30_PRO_BT:
        case USB_PRODUCT_8BITDO_PRO_2:
        case USB_PRODUCT_8BITDO_PRO_2_BT:
        case USB_PRODUCT_8BITDO_PRO_3:
        case USB_PRODUCT_8BITDO_ULTIMATE2_WIRELESS:
            return true;
        default:
            break;
        }
    }
    return false;
}

static bool HIDAPI_Driver8BitDo_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_Driver8BitDo_Context *ctx = (SDL_Driver8BitDo_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    device->context = ctx;

    if (device->product_id == USB_PRODUCT_8BITDO_ULTIMATE2_WIRELESS) {
        // The Ultimate 2 Wireless v1.02 firmware has 12 byte reports, v1.03 firmware has 34 byte reports
        const int ULTIMATE2_WIRELESS_V103_REPORT_SIZE = 34;
        const int MAX_ATTEMPTS = 3;

        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            Uint8 data[USB_PACKET_LENGTH];
            int size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 80);
            if (size == 0) {
                // Try again
                continue;
            }
            if (size >= ULTIMATE2_WIRELESS_V103_REPORT_SIZE) {
                ctx->sensors_supported = true;
                ctx->rumble_supported = true;
                ctx->powerstate_supported = true;
            }
            break;
        }
    } else {
        Uint8 data[USB_PACKET_LENGTH];
        const int MAX_ATTEMPTS = 5;
        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            int size = ReadFeatureReport(device->dev, SDL_8BITDO_FEATURE_REPORTID_ENABLE_SDL_REPORTID, data, sizeof(data));
            if (size > 0) {
#ifdef DEBUG_8BITDO_PROTOCOL
                HIDAPI_DumpPacket("8BitDo features packet: size = %d", data, size);
#endif
                ctx->sensors_supported = true;
                ctx->rumble_supported = true;
                ctx->powerstate_supported = true;

                if (size >= 14 && data[13] == SDL_8BITDO_SENSOR_TIMESTAMP_ENABLE) {
                    ctx->sensor_timestamp_supported = true;
                }

                // Set the serial number to the Bluetooth MAC address
                if (size >= 12 && data[10] != 0) {
                    char serial[18];
                    (void)SDL_snprintf(serial, sizeof(serial), "%.2x-%.2x-%.2x-%.2x-%.2x-%.2x",
                                       data[10], data[9], data[8], data[7], data[6], data[5]);
                    HIDAPI_SetDeviceSerial(device, serial);
                }
                break;
            }

            // Try again
            SDL_Delay(10);
        }
    }

    if (device->product_id == USB_PRODUCT_8BITDO_SF30_PRO || device->product_id == USB_PRODUCT_8BITDO_SF30_PRO_BT) {
        HIDAPI_SetDeviceName(device, "8BitDo SF30 Pro");
    } else if (device->product_id == USB_PRODUCT_8BITDO_SN30_PRO || device->product_id == USB_PRODUCT_8BITDO_SN30_PRO_BT) {
        HIDAPI_SetDeviceName(device, "8BitDo SN30 Pro");
    } else if (device->product_id == USB_PRODUCT_8BITDO_PRO_2 || device->product_id == USB_PRODUCT_8BITDO_PRO_2_BT) {
        HIDAPI_SetDeviceName(device, "8BitDo Pro 2");
    } else if (device->product_id == USB_PRODUCT_8BITDO_PRO_3) {
        HIDAPI_SetDeviceName(device, "8BitDo Pro 3");
    }

    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_Driver8BitDo_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_Driver8BitDo_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static Uint64 HIDAPI_Driver8BitDo_GetIMURateForProductID(SDL_HIDAPI_Device *device)
{
    SDL_Driver8BitDo_Context *ctx = (SDL_Driver8BitDo_Context *)device->context;

    // TODO: If sensor time stamp is sent, these fixed settings from observation can be replaced
    switch (device->product_id) {
    case USB_PRODUCT_8BITDO_SF30_PRO:
    case USB_PRODUCT_8BITDO_SF30_PRO_BT:
    case USB_PRODUCT_8BITDO_SN30_PRO:
    case USB_PRODUCT_8BITDO_SN30_PRO_BT:
        if (device->is_bluetooth) {
            // Note, This is estimated by observation of Bluetooth packets received in the testcontroller tool
            return 70; // Observed to be anywhere between 60-90 hz. Possibly lossy in current state
        } else if (ctx->sensor_timestamp_supported) {
            // This firmware appears to update at 200 Hz over USB
            return 200;
        } else {
            // This firmware appears to update at 100 Hz over USB
            return 100;
        }
    case USB_PRODUCT_8BITDO_PRO_2:
    case USB_PRODUCT_8BITDO_PRO_2_BT: // Note, labeled as "BT" but appears this way when wired.
    case USB_PRODUCT_8BITDO_PRO_3:
        if (device->is_bluetooth) {
            // Note, This is estimated by observation of Bluetooth packets received in the testcontroller tool
            return 85; // Observed Bluetooth packet rate seems to be 80-90hz
        } else if (ctx->sensor_timestamp_supported) {
            // This firmware appears to update at 200 Hz over USB
            return 200;
        } else {
            // This firmware appears to update at 100 Hz over USB
            return 100;
        }
    case USB_PRODUCT_8BITDO_ULTIMATE2_WIRELESS:
        if (device->is_bluetooth) {
            // Note, This is estimated by observation of Bluetooth packets received in the testcontroller tool
            return 120; // Observed Bluetooth packet rate seems to be 120hz
        } else {
            // This firmware appears to update at 1000 Hz over USB dongle
            return 1000;
        }
    default:
        return 120;
    }
}

#ifndef DEG2RAD
#define DEG2RAD(x) ((float)(x) * (float)(SDL_PI_F / 180.f))
#endif

static bool HIDAPI_Driver8BitDo_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_Driver8BitDo_Context *ctx = (SDL_Driver8BitDo_Context *)device->context;

    SDL_AssertJoysticksLocked();

    SDL_zeroa(ctx->last_state);

    // Initialize the joystick capabilities
    if (device->product_id == USB_PRODUCT_8BITDO_PRO_2 ||
        device->product_id == USB_PRODUCT_8BITDO_PRO_2_BT ||
        device->product_id == USB_PRODUCT_8BITDO_PRO_3 ||
        device->product_id == USB_PRODUCT_8BITDO_ULTIMATE2_WIRELESS) {
        // This controller has additional buttons
        joystick->nbuttons = SDL_GAMEPAD_NUM_8BITDO_BUTTONS;
    } else {
        joystick->nbuttons = 11;
    }
    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
    joystick->nhats = 1;

    if (ctx->sensors_supported) {

        // Different 8Bitdo controllers in different connection modes have different polling rates.
        const Uint64 imu_polling_rate = HIDAPI_Driver8BitDo_GetIMURateForProductID(device);
        ctx->sensor_timestamp_interval = SDL_NS_PER_SECOND / imu_polling_rate;

        SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO, (float)imu_polling_rate);
        SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, (float)imu_polling_rate);

        ctx->accelScale = SDL_STANDARD_GRAVITY / ABITDO_ACCEL_SCALE;
        // Hardware senses +/- N Degrees per second mapped to +/- INT16_MAX
        ctx->gyroScale = DEG2RAD(ABITDO_GYRO_MAX_DEGREES_PER_SECOND) / INT16_MAX;
    }

    return true;
}

static bool HIDAPI_Driver8BitDo_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_Driver8BitDo_Context *ctx = (SDL_Driver8BitDo_Context *)device->context;
    if (ctx->rumble_supported) {
        Uint8 rumble_packet[5] = { 0x05, 0x00, 0x00, 0x00, 0x00 };
        rumble_packet[1] = low_frequency_rumble >> 8;
        rumble_packet[2] = high_frequency_rumble >> 8;

        if (SDL_HIDAPI_SendRumble(device, rumble_packet, sizeof(rumble_packet)) != sizeof(rumble_packet)) {
            return SDL_SetError("Couldn't send rumble packet");
        }
        return true;
    } else {
        return SDL_Unsupported();
    }
}

static bool HIDAPI_Driver8BitDo_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_Driver8BitDo_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_Driver8BitDo_Context *ctx = (SDL_Driver8BitDo_Context *)device->context;
    Uint32 caps = 0;
    if (ctx->rumble_supported) {
        caps |= SDL_JOYSTICK_CAP_RUMBLE;
    }
    if (ctx->rgb_supported) {
        caps |= SDL_JOYSTICK_CAP_RGB_LED;
    }
    return caps;
}

static bool HIDAPI_Driver8BitDo_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_Driver8BitDo_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool HIDAPI_Driver8BitDo_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    SDL_Driver8BitDo_Context *ctx = (SDL_Driver8BitDo_Context *)device->context;
    if (ctx->sensors_supported) {
        ctx->sensors_enabled = enabled;
        return true;
    }
    return SDL_Unsupported();
}

static void HIDAPI_Driver8BitDo_HandleOldStatePacket(SDL_Joystick *joystick, SDL_Driver8BitDo_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

    if (ctx->last_state[2] != data[2]) {
        Uint8 hat;

        switch (data[2]) {
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

    if (ctx->last_state[0] != data[0]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[0] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[0] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[0] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[0] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[0] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[0] & 0x80) != 0));
    }

    if (ctx->last_state[1] != data[1]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[1] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[1] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[1] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[1] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[1] & 0x40) != 0));

        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, (data[1] & 0x01) ? SDL_MAX_SINT16 : SDL_MIN_SINT16);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, (data[1] & 0x02) ? SDL_MAX_SINT16 : SDL_MIN_SINT16);
    }

#define READ_STICK_AXIS(offset) \
    (data[offset] == 0x7f ? 0 : (Sint16)HIDAPI_RemapVal((float)((int)data[offset] - 0x7f), -0x7f, 0xff - 0x7f, SDL_MIN_SINT16, SDL_MAX_SINT16))
    {
        axis = READ_STICK_AXIS(3);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
        axis = READ_STICK_AXIS(4);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
        axis = READ_STICK_AXIS(5);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
        axis = READ_STICK_AXIS(6);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);
    }
#undef READ_STICK_AXIS

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static void HIDAPI_Driver8BitDo_HandleStatePacket(SDL_Joystick *joystick, SDL_Driver8BitDo_Context *ctx, Uint8 *data, int size)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

    switch (data[0]) {
    case SDL_8BITDO_REPORTID_NOT_SUPPORTED_SDL_REPORTID:    // Firmware without enhanced mode
    case SDL_8BITDO_REPORTID_SDL_REPORTID:                  // Enhanced mode USB report
    case SDL_8BITDO_BT_REPORTID_SDL_REPORTID:               // Enhanced mode Bluetooth report
        break;
    default:
        // We don't know how to handle this report
        return;
    }

    if (ctx->last_state[1] != data[1]) {
        Uint8 hat;

        switch (data[1]) {
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


    if (ctx->last_state[8] != data[8]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data[8] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data[8] & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data[8] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data[8] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data[8] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data[8] & 0x80) != 0));

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_8BITDO_PL, ((data[8] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_8BITDO_PR, ((data[8] & 0x04) != 0));
    }

    if (ctx->last_state[9] != data[9]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data[9] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data[9] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data[9] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data[9] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data[9] & 0x40) != 0));
    }

    if (size > 10 && ctx->last_state[10] != data[10]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_8BITDO_L4, ((data[10] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_8BITDO_R4, ((data[10] & 0x02) != 0));
    }

#define READ_STICK_AXIS(offset) \
    (data[offset] == 0x7f ? 0 : (Sint16)HIDAPI_RemapVal((float)((int)data[offset] - 0x7f), -0x7f, 0xff - 0x7f, SDL_MIN_SINT16, SDL_MAX_SINT16))
    {
        axis = READ_STICK_AXIS(2);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
        axis = READ_STICK_AXIS(3);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
        axis = READ_STICK_AXIS(4);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
        axis = READ_STICK_AXIS(5);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);
    }
#undef READ_STICK_AXIS

#define READ_TRIGGER_AXIS(offset) \
    (Sint16)(((int)data[offset] * 257) - 32768)
    {
        axis = READ_TRIGGER_AXIS(7);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
        axis = READ_TRIGGER_AXIS(6);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);
    }
#undef READ_TRIGGER_AXIS

    if (ctx->powerstate_supported) {
        SDL_PowerState state;
        int percent;
        Uint8 status = data[14] >> 7;
        Uint8 level = (data[14] & 0x7f);
        if (level == 100) {
            status = 2;
        }
        switch (status) {
        case 0:
            state = SDL_POWERSTATE_ON_BATTERY;
            percent = level;
            break;
        case 1:
            state = SDL_POWERSTATE_CHARGING;
            percent = level;
            break;
        case 2:
            state = SDL_POWERSTATE_CHARGED;
            percent = 100;
            break;
        default:
            state = SDL_POWERSTATE_UNKNOWN;
            percent = 0;
            break;
        }
        SDL_SendJoystickPowerInfo(joystick, state, percent);
    }

    if (ctx->sensors_enabled) {
        Uint64 sensor_timestamp;
        float values[3];
        ABITDO_SENSORS *sensors = (ABITDO_SENSORS *)&data[15];

         if (ctx->sensor_timestamp_supported) {
            Uint32 delta;
            Uint32 tick = LOAD32(data[27], data[28], data[29], data[30]);

            if (ctx->last_tick) {
                if (ctx->last_tick < tick) {
                    delta = (tick - ctx->last_tick);
                } else {
                    delta = (SDL_MAX_UINT32 - ctx->last_tick + tick + 1);
                }
                // Sanity check the delta value
                if (delta < 100000) {
                    ctx->sensor_timestamp_interval = SDL_US_TO_NS(delta);
                }
            }
            ctx->last_tick = tick;
        }

        // Note: we cannot use the time stamp of the receiving computer due to packet delay creating "spiky" timings.
        // The imu time stamp is intended to be the sample time of the on-board hardware.
        // In the absence of time stamp data from the data[], we can simulate that by
        // advancing a time stamp by the observed/known imu clock rate. This is 8ms = 125 Hz
        sensor_timestamp = ctx->sensor_timestamp;
        ctx->sensor_timestamp += ctx->sensor_timestamp_interval;

        // This device's IMU values are reported differently from SDL
        // Thus we perform a rotation of the coordinate system to match the SDL standard.

        // By observation of this device:
        // Hardware x is reporting roll (rotation about the power jack's axis)
        // Hardware y is reporting pitch (rotation about the horizontal axis)
        // Hardware z is reporting yaw (rotation about the joysticks' center axis)
        values[0] = -sensors->sGyroY * ctx->gyroScale;  // Rotation around pitch axis
        values[1] = sensors->sGyroZ * ctx->gyroScale;   // Rotation around yaw axis
        values[2] = -sensors->sGyroX * ctx->gyroScale;  // Rotation around roll axis
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, sensor_timestamp, values, 3);

        // By observation of this device:
        // Accelerometer X is positive when front of the controller points toward the sky.
        // Accelerometer y is positive when left side of the controller points toward the sky.
        // Accelerometer Z is positive when sticks point toward the sky.
        values[0] = -sensors->sAccelY * ctx->accelScale; // Acceleration along pitch axis
        values[1] = sensors->sAccelZ * ctx->accelScale;  // Acceleration along yaw axis
        values[2] = -sensors->sAccelX * ctx->accelScale; // Acceleration along roll axis
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, sensor_timestamp, values, 3);
    }

    SDL_memcpy(ctx->last_state, data, SDL_min(size, sizeof(ctx->last_state)));
}

static bool HIDAPI_Driver8BitDo_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_Driver8BitDo_Context *ctx = (SDL_Driver8BitDo_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH];
    int size = 0;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    } else {
        return false;
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_8BITDO_PROTOCOL
        HIDAPI_DumpPacket("8BitDo packet: size = %d", data, size);
#endif
        if (!joystick) {
            continue;
        }

        if (size == 9) {
            // Old firmware USB report for the SF30 Pro and SN30 Pro controllers
            HIDAPI_Driver8BitDo_HandleOldStatePacket(joystick, ctx, data, size);
        } else {
            HIDAPI_Driver8BitDo_HandleStatePacket(joystick, ctx, data, size);
        }
    }

    if (size < 0) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_Driver8BitDo_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
}

static void HIDAPI_Driver8BitDo_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_Driver8BitDo = {
    SDL_HINT_JOYSTICK_HIDAPI_8BITDO,
    true,
    HIDAPI_Driver8BitDo_RegisterHints,
    HIDAPI_Driver8BitDo_UnregisterHints,
    HIDAPI_Driver8BitDo_IsEnabled,
    HIDAPI_Driver8BitDo_IsSupportedDevice,
    HIDAPI_Driver8BitDo_InitDevice,
    HIDAPI_Driver8BitDo_GetDevicePlayerIndex,
    HIDAPI_Driver8BitDo_SetDevicePlayerIndex,
    HIDAPI_Driver8BitDo_UpdateDevice,
    HIDAPI_Driver8BitDo_OpenJoystick,
    HIDAPI_Driver8BitDo_RumbleJoystick,
    HIDAPI_Driver8BitDo_RumbleJoystickTriggers,
    HIDAPI_Driver8BitDo_GetJoystickCapabilities,
    HIDAPI_Driver8BitDo_SetJoystickLED,
    HIDAPI_Driver8BitDo_SendJoystickEffect,
    HIDAPI_Driver8BitDo_SetJoystickSensorsEnabled,
    HIDAPI_Driver8BitDo_CloseJoystick,
    HIDAPI_Driver8BitDo_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_8BITDO

#endif // SDL_JOYSTICK_HIDAPI
